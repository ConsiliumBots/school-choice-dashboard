// Set working directory
cd "C:\Users\ignaciolepe\Documents\GitHub\school-choice-dashboard\data\inputs\2025"

// Import application data
import delimited "applications3.csv", clear

// Check how many unique students are in the data
unique student_id

// Keep only relevant variables (including choice_rank)
keep student_id program_id unmatched assigned choice_rank

// Indicator for whether student was assigned to any program
bysort student_id: egen any_assigned = max(assigned)

// Temporary variable: choice_rank where assigned == 1
gen assigned_rank = .
replace assigned_rank = choice_rank if assigned == 1

// Get final assigned rank per student
bysort student_id (assigned): egen final_rank = max(assigned_rank)

// Collapse to one row per student: max(any_assigned), first unmatched, max(final_rank)
collapse (max) any_assigned final_rank (first) unmatched, by(student_id)

// Set final_rank to missing if student wasn't assigned anywhere
replace final_rank = . if any_assigned == 0

tab final_rank 

// Save collapsed data
export delimited "student_assignment_summary.csv", replace
