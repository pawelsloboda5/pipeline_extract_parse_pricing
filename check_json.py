import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
from jsonschema import validate, ValidationError, Draft7Validator
import logging
from rich.console import Console
from rich.table import Table
from copy import deepcopy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class PricingSchemaValidator:
    """Validates and transforms pricing data for frontend use"""
    
    def __init__(self, schema_path: str = "schema.json"):
        self.schema = self._load_schema(schema_path)
        self.validator = Draft7Validator(self.schema)
    
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load the JSON schema from file"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {str(e)}")
            raise

    def _get_default_structure(self) -> Dict[str, Any]:
        """Get default structure for required objects"""
        return {
            "storage": {
                "amount": 0.0,
                "unit": "GB"
            },
            "api": {
                "requests": {
                    "rate": 0.0,
                    "period": "second",
                    "quota": 0.0
                }
            },
            "compute": {
                "vcpu": 0.0,
                "memory": 0.0,
                "unit": "GB"
            }
        }

    def transform_for_frontend(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data into frontend-friendly format"""
        transformed = deepcopy(data)

        def try_numeric_conversion(value: str) -> Union[float, str]:
            """Convert string to number if possible"""
            try:
                if isinstance(value, str) and value.replace('.', '').isdigit():
                    return float(value)
                return value
            except (ValueError, TypeError):
                return value

        def standardize_values(obj: Dict[str, Any]) -> None:
            """Standardize values to appropriate types"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "value":  # Special handling for value fields
                        obj[key] = try_numeric_conversion(value)
                    elif isinstance(value, dict):
                        standardize_values(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                standardize_values(item)

        def standardize_numbers(obj: Dict[str, Any]) -> None:
            """Standardize numerical fields to use null"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['amount', 'min', 'max', 'rate', 'quota']:
                        if value is None or value == "":
                            obj[key] = None  # Use null for missing numerical values
                        elif isinstance(value, str):
                            obj[key] = try_numeric_conversion(value)
                    elif isinstance(value, (dict, list)):
                        standardize_numbers(value)

        def standardize_strings(obj: Dict[str, Any]) -> None:
            """Standardize string fields to use empty string"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['description', 'logo_url', 'comments', 'name']:
                        if value is None:
                            obj[key] = ""
                    elif isinstance(value, (dict, list)):
                        standardize_strings(value)

        def standardize_arrays(obj: Dict[str, Any]) -> None:
            """Standardize arrays to never be null"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['use_cases', 'discounts', 'custom_features', 'tags']:
                        if value is None:
                            obj[key] = []
                    elif isinstance(value, (dict, list)):
                        standardize_arrays(value)

        def standardize_trial(obj: Dict[str, Any]) -> None:
            """Standardize trial information"""
            if "trial" not in obj or obj["trial"] is None:
                obj["trial"] = {
                    "available": False,
                    "duration_days": None,
                    "auto_renewal": None,
                    "requires_credit_card": None,
                    "restrictions": []
                }
            else:
                trial = obj["trial"]
                trial["duration_days"] = try_numeric_conversion(trial.get("duration_days"))
                trial["auto_renewal"] = trial.get("auto_renewal", None)
                trial["requires_credit_card"] = trial.get("requires_credit_card", None)
                trial["restrictions"] = trial.get("restrictions", [])

        def standardize_enterprise(obj: Dict[str, Any]) -> None:
            """Standardize enterprise information"""
            if "enterprise" not in obj or obj["enterprise"] is None:
                obj["enterprise"] = {
                    "available": False,
                    "contact_sales": False,
                    "minimum_seats": None,
                    "minimum_contract_length_months": None,
                    "custom_features": [],
                    "security_features": [],
                    "support_level": {
                        "type": "",
                        "response_time": "",
                        "availability": "",
                        "includes_dedicated_support": False
                    }
                }
            else:
                enterprise = obj["enterprise"]
                if "support_level" not in enterprise:
                    enterprise["support_level"] = {
                        "type": "",
                        "response_time": "",
                        "availability": "",
                        "includes_dedicated_support": False
                    }
                enterprise["security_features"] = enterprise.get("security_features", [])
                enterprise["minimum_contract_length_months"] = try_numeric_conversion(
                    enterprise.get("minimum_contract_length_months")
                )

        def standardize_feature_details(obj: Dict[str, Any]) -> None:
            """Standardize feature details"""
            if isinstance(obj, dict):
                if "features" in obj and isinstance(obj["features"], dict):
                    for category in obj["features"].get("categories", []):
                        for feature in category.get("features", []):
                            if "limitations" not in feature:
                                feature["limitations"] = []
                            if "availability" not in feature:
                                feature["availability"] = "All plans"
            
                for value in obj.values():
                    if isinstance(value, dict):
                        standardize_feature_details(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                standardize_feature_details(item)

        def standardize_addons(obj: Dict[str, Any]) -> None:
            """Standardize add-ons information"""
            if "add_ons" not in obj or obj["add_ons"] is None:
                obj["add_ons"] = []
            else:
                for addon in obj["add_ons"]:
                    if "pricing" not in addon or addon["pricing"] is None:
                        addon["pricing"] = {
                            "amount": None,
                            "period": "month",
                            "currency": "USD",
                            "is_per_user": False,
                            "amount_range": None,
                            "conditions": []
                        }
                    else:
                        pricing = addon["pricing"]
                        pricing["amount"] = try_numeric_conversion(pricing.get("amount"))
                        pricing["is_per_user"] = pricing.get("is_per_user", False)
                        pricing["conditions"] = pricing.get("conditions", [])
                        
                    addon["availability"] = addon.get("availability", [])

        # Add new standardization to the transformation pipeline
        standardize_enterprise(transformed)
        standardize_feature_details(transformed)
        standardize_trial(transformed)
        standardize_addons(transformed)
        
        # Existing transformations...
        standardize_values(transformed)
        standardize_numbers(transformed)
        standardize_strings(transformed)
        standardize_arrays(transformed)

        return transformed

    def validate_and_transform(self, data: Union[str, Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate and transform data"""
        try:
            # If data is a string (file path), load it
            if isinstance(data, str):
                with open(data, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # Transform data
            transformed_data = self.transform_for_frontend(data)

            # Validate transformed data
            errors = []
            try:
                validate(instance=transformed_data, schema=self.schema)
                is_valid = True
            except ValidationError as e:
                errors.append(f"Validation error at {' -> '.join(str(p) for p in e.path)}: {e.message}")
                is_valid = False

            return is_valid, transformed_data, errors

        except Exception as e:
            logger.error(f"Failed to process data: {str(e)}")
            return False, {}, [str(e)]

    def save_transformed(self, data: Dict[str, Any], output_path: str) -> None:
        """Save transformed data to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def main():
    """Main function to run validation and transformation"""
    validator = PricingSchemaValidator()
    
    # Example usage
    json_path = "extracted_info/parsed_calendly_20241229_183540.json"
    output_path = "extracted_info/transformed_calendly_20241229_183540.json"
    
    try:
        is_valid, transformed_data, errors = validator.validate_and_transform(json_path)
        
        if errors:
            console.print("[red]Validation Errors:[/red]")
            for error in errors:
                console.print(f"[red]• {error}[/red]")
        
        if transformed_data:
            validator.save_transformed(transformed_data, output_path)
            console.print(f"[green]Transformed data saved to: {output_path}[/green]")
            
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")

if __name__ == "__main__":
    main()
