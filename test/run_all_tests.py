#!/usr/bin/env python3
"""
GestureBot Test Suite Runner
Comprehensive test runner for all GestureBot vision system components.
"""

import os
import sys
import time
import json
import subprocess
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestSuiteResult:
    """Test suite execution result."""
    test_name: str
    success: bool
    execution_time: float
    tests_run: int
    failures: int
    errors: int
    skipped: int
    error_details: List[str] = None
    
    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []


class GestureBotTestRunner:
    """Main test runner for GestureBot vision system."""
    
    def __init__(self):
        self.package_dir = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        self.results_dir = self.test_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Test modules to run
        self.test_modules = [
            'test_mediapipe_models',
            'test_camera_integration', 
            'test_vision_nodes'
        ]
        
        self.test_results = []
        
        print("=== GestureBot Vision System Test Suite ===")
        print(f"Package directory: {self.package_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Results directory: {self.results_dir}")
    
    def run_individual_test(self, test_module: str) -> TestSuiteResult:
        """Run an individual test module."""
        print(f"\n🧪 Running {test_module}...")
        
        start_time = time.time()
        
        try:
            # Import and run the test module
            test_file = self.test_dir / f"{test_module}.py"
            
            if not test_file.exists():
                return TestSuiteResult(
                    test_name=test_module,
                    success=False,
                    execution_time=0,
                    tests_run=0,
                    failures=0,
                    errors=1,
                    skipped=0,
                    error_details=[f"Test file not found: {test_file}"]
                )
            
            # Run test using subprocess to isolate each test
            cmd = [sys.executable, str(test_file)]
            result = subprocess.run(
                cmd,
                cwd=str(self.package_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test
            )
            
            execution_time = time.time() - start_time
            
            # Parse test results from output
            tests_run, failures, errors, skipped = self.parse_unittest_output(result.stderr)
            
            success = result.returncode == 0 and failures == 0 and errors == 0
            
            error_details = []
            if not success:
                error_details.append(f"Return code: {result.returncode}")
                if result.stdout:
                    error_details.append(f"STDOUT: {result.stdout}")
                if result.stderr:
                    error_details.append(f"STDERR: {result.stderr}")
            
            test_result = TestSuiteResult(
                test_name=test_module,
                success=success,
                execution_time=execution_time,
                tests_run=tests_run,
                failures=failures,
                errors=errors,
                skipped=skipped,
                error_details=error_details
            )
            
            # Print results
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} - {tests_run} tests, {execution_time:.1f}s")
            
            if not success:
                print(f"   Failures: {failures}, Errors: {errors}, Skipped: {skipped}")
                if error_details:
                    print(f"   Error details: {error_details[0][:100]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestSuiteResult(
                test_name=test_module,
                success=False,
                execution_time=execution_time,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                error_details=[f"Test timed out after {execution_time:.1f}s"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestSuiteResult(
                test_name=test_module,
                success=False,
                execution_time=execution_time,
                tests_run=0,
                failures=0,
                errors=1,
                skipped=0,
                error_details=[f"Test runner error: {str(e)}"]
            )
    
    def parse_unittest_output(self, output: str) -> tuple:
        """Parse unittest output to extract test statistics."""
        tests_run = 0
        failures = 0
        errors = 0
        skipped = 0
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for unittest summary line
            if 'Ran' in line and 'test' in line:
                try:
                    # Extract number of tests run
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Ran':
                            tests_run = int(parts[i + 1])
                            break
                except (ValueError, IndexError):
                    pass
            
            # Look for failure/error counts
            if 'FAILED' in line:
                try:
                    # Parse "FAILED (failures=X, errors=Y)" format
                    if 'failures=' in line:
                        failures_part = line.split('failures=')[1].split(',')[0].split(')')[0]
                        failures = int(failures_part)
                    if 'errors=' in line:
                        errors_part = line.split('errors=')[1].split(',')[0].split(')')[0]
                        errors = int(errors_part)
                    if 'skipped=' in line:
                        skipped_part = line.split('skipped=')[1].split(',')[0].split(')')[0]
                        skipped = int(skipped_part)
                except (ValueError, IndexError):
                    pass
        
        return tests_run, failures, errors, skipped
    
    def run_all_tests(self) -> bool:
        """Run all test modules."""
        print(f"\n🚀 Starting test suite with {len(self.test_modules)} test modules...")
        
        overall_start_time = time.time()
        
        for test_module in self.test_modules:
            result = self.run_individual_test(test_module)
            self.test_results.append(result)
        
        overall_execution_time = time.time() - overall_start_time
        
        # Generate summary
        self.generate_test_summary(overall_execution_time)
        
        # Save detailed results
        self.save_test_results()
        
        # Return overall success
        return all(result.success for result in self.test_results)
    
    def generate_test_summary(self, total_time: float):
        """Generate and print test summary."""
        print("\n" + "="*60)
        print("GESTUREBOT VISION SYSTEM TEST SUMMARY")
        print("="*60)
        
        total_tests = sum(r.tests_run for r in self.test_results)
        total_failures = sum(r.failures for r in self.test_results)
        total_errors = sum(r.errors for r in self.test_results)
        total_skipped = sum(r.skipped for r in self.test_results)
        successful_modules = sum(1 for r in self.test_results if r.success)
        
        print(f"Test Modules: {len(self.test_modules)}")
        print(f"Successful Modules: {successful_modules}")
        print(f"Failed Modules: {len(self.test_modules) - successful_modules}")
        print(f"Total Execution Time: {total_time:.1f}s")
        print()
        print(f"Total Tests Run: {total_tests}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Success Rate: {((total_tests - total_failures - total_errors) / max(total_tests, 1) * 100):.1f}%")
        
        print("\nDETAILED RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"  {status} {result.test_name}: {result.tests_run} tests in {result.execution_time:.1f}s")
            
            if not result.success and result.error_details:
                print(f"    └─ {result.error_details[0][:80]}...")
        
        if all(r.success for r in self.test_results):
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print("\n⚠️  SOME TESTS FAILED - CHECK DETAILS ABOVE")
    
    def save_test_results(self):
        """Save detailed test results to JSON file."""
        timestamp = int(time.time())
        results_file = self.results_dir / f'test_results_{timestamp}.json'
        
        results_data = {
            'timestamp': timestamp,
            'package': 'gesturebot',
            'test_suite': 'vision_system',
            'results': [asdict(result) for result in self.test_results],
            'summary': {
                'total_modules': len(self.test_modules),
                'successful_modules': sum(1 for r in self.test_results if r.success),
                'total_tests': sum(r.tests_run for r in self.test_results),
                'total_failures': sum(r.failures for r in self.test_results),
                'total_errors': sum(r.errors for r in self.test_results),
                'total_skipped': sum(r.skipped for r in self.test_results)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met for testing."""
        print("\n🔍 Checking test prerequisites...")
        
        prerequisites_met = True
        
        # Check Python packages
        required_packages = ['cv2', 'numpy', 'rclpy', 'mediapipe']
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package} available")
            except ImportError:
                print(f"  ❌ {package} not available")
                prerequisites_met = False
        
        # Check ROS 2 environment
        if 'ROS_DISTRO' in os.environ:
            print(f"  ✅ ROS 2 {os.environ['ROS_DISTRO']} environment detected")
        else:
            print("  ⚠️  ROS 2 environment not detected")
            prerequisites_met = False
        
        # Check package structure
        required_dirs = ['src', 'msg', 'srv', 'launch']
        for dir_name in required_dirs:
            dir_path = self.package_dir / dir_name
            if dir_path.exists():
                print(f"  ✅ {dir_name}/ directory found")
            else:
                print(f"  ❌ {dir_name}/ directory missing")
                prerequisites_met = False
        
        return prerequisites_met


def main():
    """Main function for test runner."""
    runner = GestureBotTestRunner()
    
    # Check prerequisites
    if not runner.check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required packages and set up ROS 2 environment.")
        return 1
    
    # Run all tests
    success = runner.run_all_tests()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
