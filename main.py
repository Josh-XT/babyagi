#!/usr/bin/env python3
import argparse
from TMS import TaskManagementSystem

def main(primary_objective):
    tms = TaskManagementSystem(primary_objective=primary_objective)
    tms.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Management System")
    parser.add_argument("primary_objective", help="Specify the primary objective for the Task Management System")

    args = parser.parse_args()
    main(args.primary_objective)
