"""
Project pricing module.

Market pricing estimates for MMM consulting projects.
"""

from __future__ import annotations


# Component pricing based on market research (USD)
PROJECT_COMPONENTS = {
    "data_pipeline": {
        "name": "Data Pipeline & ETL",
        "price": 5000,
        "description": "Data collection, cleaning, and transformation",
    },
    "eda": {
        "name": "Exploratory Data Analysis",
        "price": 3000,
        "description": "Statistical analysis and visualization",
    },
    "baseline_model": {
        "name": "Baseline Model (Ridge)",
        "price": 5000,
        "description": "Frequentist baseline for comparison",
    },
    "bayesian_model": {
        "name": "Bayesian MMM",
        "price": 15000,
        "description": "PyMC-Marketing implementation with uncertainty",
    },
    "hierarchical_extension": {
        "name": "Hierarchical Multi-Region",
        "price": 10000,
        "description": "Multi-region analysis with partial pooling",
    },
    "optimization": {
        "name": "Budget Optimization",
        "price": 8000,
        "description": "Optimal allocation and marginal ROAS",
    },
    "dashboard": {
        "name": "Interactive Dashboard",
        "price": 7000,
        "description": "Streamlit dashboard with visualizations",
    },
    "documentation": {
        "name": "Documentation & Training",
        "price": 3000,
        "description": "Technical docs and user training",
    },
    "support_1month": {
        "name": "Support (1 month)",
        "price": 2000,
        "description": "Post-delivery support and adjustments",
    },
}

# Package definitions
PACKAGES = {
    "basic": {
        "name": "Basic MMM",
        "components": ["data_pipeline", "eda", "baseline_model", "documentation"],
        "discount_pct": 10,
    },
    "standard": {
        "name": "Standard Bayesian MMM",
        "components": [
            "data_pipeline", "eda", "baseline_model", "bayesian_model",
            "optimization", "documentation",
        ],
        "discount_pct": 15,
    },
    "enterprise": {
        "name": "Enterprise Multi-Region MMM",
        "components": list(PROJECT_COMPONENTS.keys()),
        "discount_pct": 20,
    },
}


def get_component_price(component: str) -> float:
    """Get price for a single component."""
    return PROJECT_COMPONENTS.get(component, {}).get("price", 0)


def estimate_project_price(
    components: list[str] | None = None,
    package: str | None = None,
    discount_pct: float = 0,
) -> dict:
    """
    Estimate project market price.

    Args:
        components: List of component keys to include.
        package: Optional package name (basic, standard, enterprise).
        discount_pct: Custom discount percentage.

    Returns:
        Dict with pricing breakdown.
    """
    if package and package in PACKAGES:
        pkg = PACKAGES[package]
        components = pkg["components"]
        discount_pct = pkg["discount_pct"]
        package_name = pkg["name"]
    else:
        package_name = "Custom"

    if components is None:
        components = list(PROJECT_COMPONENTS.keys())

    # Calculate component prices
    component_details = []
    for comp in components:
        if comp in PROJECT_COMPONENTS:
            info = PROJECT_COMPONENTS[comp]
            component_details.append({
                "key": comp,
                "name": info["name"],
                "price": info["price"],
                "description": info["description"],
            })

    subtotal = sum(c["price"] for c in component_details)
    discount_amount = subtotal * (discount_pct / 100)
    total = subtotal - discount_amount

    return {
        "package": package_name,
        "components": component_details,
        "subtotal": subtotal,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "total": total,
    }


def get_market_comparison() -> list[dict]:
    """
    Get market comparison data for MMM projects.

    Returns:
        List of competitor pricing data.
    """
    return [
        {
            "provider": "Traditional Consulting",
            "type": "Full-service agency",
            "price_range": "$50,000 - $100,000+",
            "timeline": "3-6 months",
        },
        {
            "provider": "Boutique Analytics",
            "type": "Specialized MMM firm",
            "price_range": "$25,000 - $50,000",
            "timeline": "6-12 weeks",
        },
        {
            "provider": "SaaS Platform",
            "type": "Self-service tool",
            "price_range": "$2,000 - $5,000/month",
            "timeline": "2-4 weeks setup",
        },
        {
            "provider": "This Project",
            "type": "Custom Bayesian MMM",
            "price_range": "$40,000 - $60,000",
            "timeline": "4-8 weeks",
        },
    ]


def calculate_cost_savings(
    traditional_cost: float = 75000,
    this_project_cost: float = 50000,
) -> dict:
    """
    Calculate cost savings vs traditional consulting.

    Returns:
        Dict with savings metrics.
    """
    savings = traditional_cost - this_project_cost
    savings_pct = (savings / traditional_cost) * 100

    return {
        "traditional_cost": traditional_cost,
        "this_project_cost": this_project_cost,
        "savings_amount": savings,
        "savings_pct": savings_pct,
    }
