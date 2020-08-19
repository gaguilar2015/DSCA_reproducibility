# Review of branch activity-3-report
RLe

I followed the script and can see the use case for this form of reporting. A couple of observations:

* preprocessing.R line 27 onwards- in investigating null matches,  I have not come across the method you use here, using str_c while specifying the delimiter to collapse by. Probably not important but it appears to be the same use case as dplyr::set_diff() or dplyr::anti_join(), though these would provide you with the null match values in one step. 
* Country_Report_step_1_2.Rmd  From line 110 on - would this data prep code chunk be moved to a separate preprocessing script to source? 
* Country_Report_step_1_2.Rmd  Line 127 - the continent_map_data contains null values due to left_join. Just checking that this is as expected. Also continent is hard-coded.
* Country_Report_step_3_4.Rmd & Country_Report_step_5.Rmd - these wouldn’t run. As the preprocessing scripts had a broken pathname.Or is that the point? I’m not 100% following the purpose of the task I guess- is it to follow on the iterations and then build a report that is closer to a finished product? Then the users finding bugs as we go? 
* I’ll include a separate .md with additional guide to the current tasks. 
