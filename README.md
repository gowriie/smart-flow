SMART FLOW
An Explainable Fuzzy Graph-Based System for Optimal Hydraulic Network Design


PROJECT OVERVIEW

Smart Flow is an intelligent pipeline routing and evaluation system designed for optimal water distribution network planning. The system integrates graph-based routing with fuzzy multi-criteria decision making to support efficient and explainable pipeline layout design.

Traditional pipeline planning approaches often rely on shortest path algorithms or manual engineering decisions. These methods do not consider multiple real-world factors simultaneously. Smart Flow addresses this limitation by combining spatial modelling, heuristic routing, and fuzzy evaluation to produce balanced and interpretable solutions.


PROBLEM STATEMENT

Designing pipeline networks in residential or urban layouts is a complex problem due to the presence of obstacles and multiple design constraints. Engineers must consider several factors such as pipeline length, installation cost, safety risk, and pressure impact.

Shortest path algorithms alone cannot provide optimal solutions because they ignore these additional factors. Therefore, a system is required that can generate multiple feasible routes and evaluate them based on multiple criteria to select the most suitable pipeline path.



OBJECTIVE

The objective of this project is to develop an integrated framework that can generate and evaluate pipeline routes using intelligent computational techniques.

The system aims to:

Generate multiple feasible pipeline routes between source and destination
Evaluate each route using fuzzy multi-criteria decision analysis
Select the most suitable route based on combined evaluation
Provide explainable output for engineering decision support



SYSTEM ARCHITECTURE

The Smart Flow system consists of the following major components:

Spatial Modelling Layer
The blueprint layout is converted into a grid-based graph structure. Each grid cell represents a node and connections between cells represent edges. Obstacle regions are marked as restricted nodes.

Routing Layer
The A star search algorithm is used to generate multiple candidate routes between source and destination while avoiding obstacles.

Evaluation Layer
Each candidate route is evaluated using fuzzy logic based on multiple parameters such as length, cost, risk, and pressure impact.

Decision Layer
A composite suitability score is calculated for each route, and the best route is selected. The system also provides explanation for the decision.


TECHNOLOGIES USED

Programming Language
Python

Libraries and Frameworks
Ultralytics YOLO for object detection
NumPy for numerical operations
Matplotlib for visualization
Streamlit for user interface

Algorithms
A star search algorithm for routing
Fuzzy logic for multi-criteria evaluation



PROJECT STRUCTURE

The project is organized into the following modules:

src/detection
Contains object detection and dataset preparation modules

src/graph_routing
Implements grid construction and path planning algorithms

src/fuzzy
Contains fuzzy evaluation logic and scoring mechanisms

src/map_generation
Generates synthetic maps and test layouts

src/simulation_ui
Implements the Streamlit-based user interface

data
Contains datasets and test inputs

runs
Stores output results and generated route visualizations


FEATURES

Supports blueprint-based spatial modelling
Generates multiple pipeline route options
Applies multi-objective decision making
Uses fuzzy logic for realistic evaluation
Provides explainable decision output
Interactive user interface using Streamlit


WORKFLOW

The system follows a structured workflow:

Input blueprint layout is processed
Grid-based graph is constructed
A star algorithm generates candidate routes
Each route is evaluated using fuzzy logic
Final best route is selected based on score
Results are visualized in the user interface


HOW TO RUN THE PROJECT

Install required dependencies

pip install -r requirements.txt

Run the application

streamlit run src/simulation_ui/app.py

The application will open in a browser interface where users can visualize routes and results.


IMPORTANT NOTE

Model weight files such as YOLO models are not included in the repository due to size constraints. These files will be automatically downloaded when required or can be added manually.



APPLICATIONS

Residential pipeline layout planning
Urban water distribution network design
Smart city infrastructure development
Engineering decision support systems



FUTURE ENHANCEMENTS

Integration with real hydraulic simulation tools
Incorporation of real-time pressure and flow data
Use of machine learning for risk prediction
Extension to large-scale urban planning systems
Integration with GIS-based mapping systems



CONCLUSION

Smart Flow provides an intelligent and explainable approach to pipeline routing by combining graph-based algorithms with fuzzy multi-objective evaluation. The system improves decision quality by considering multiple real-world factors and offering transparent reasoning for route selection.

This framework can be extended further to support advanced infrastructure planning in smart city environments.


