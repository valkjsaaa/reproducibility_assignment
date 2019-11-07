#%%
import pandas as pd
raw_data = pd.read_csv('GroupD_9-2-2015/data/materials-9859-Top-level_materials/12022-Exp1.csv', names=[
    'sub_id', 'block_id', 'trial_id',
    'color_target', 'identity_target', 'location_target',
    'color_response', 'identity_response', 'location_response',
    'color_accuracy', 'identity_accuracy', 'location_accuracy'
])

#%%
processed_data = raw_data.copy()
processed_data['presuprise'] = (processed_data.trial_id < 156)
processed_data['suprise'] = (processed_data.trial_id == 156)
processed_data['control'] = (processed_data.trial_id > 156)
processed_data['control_id'] = processed_data.trial_id - 156

#%%
presuprise_trials = processed_data[processed_data.presuprise]
presuprise_location_accuracy = presuprise_trials.location_accuracy.mean()
print(f"On the presurprise trials, {presuprise_location_accuracy * 100}% of responses "
      f"in the location task were correct")

#%%
suprise_trials = processed_data[processed_data.suprise]
suprise_color_accuracy = suprise_trials.color_accuracy.mean()
suprise_color_total = suprise_trials.color_accuracy.count()
suprise_color_correct = suprise_trials.color_accuracy.sum()
print(f"Only {suprise_color_correct} of {suprise_color_total} ({suprise_color_accuracy * 100}%) "
      f"participants correctly reported the color of the target letter")

#%%
suprise_trials = processed_data[processed_data.suprise]
suprise_identity_accuracy = suprise_trials.identity_accuracy.mean()
suprise_identity_total = suprise_trials.identity_accuracy.count()
suprise_identity_correct = suprise_trials.identity_accuracy.sum()
print(f"Furthermore, performance on the identity task "
      f"({suprise_identity_accuracy * 100}% correct) was exactly at chance level.")

#%%
suprise_trials = processed_data[processed_data.suprise]
suprise_location_accuracy = suprise_trials.location_accuracy.mean()
print(f"Participants’ performance on the location task was good ({suprise_location_accuracy * 100}% correct)")

#%%
first_control_trials = processed_data[processed_data.control_id == 1]
first_control_color_accuracy = first_control_trials.color_accuracy.mean()
first_control_color_total = first_control_trials.color_accuracy.count()
first_control_color_correct = first_control_trials.color_accuracy.sum()
first_control_identity_accuracy = first_control_trials.identity_accuracy.mean()
first_control_identity_total = first_control_trials.identity_accuracy.count()
first_control_identity_correct = first_control_trials.identity_accuracy.sum()
print(f"Participants exhibited a dramatic increase in reporting accuracy for "
      f"the target letter’s color ({first_control_color_accuracy * 100}% correct) "
      f"and identity ({first_control_identity_accuracy * 100}% correct) on the first control trial")

#%%
last_control_trials = [processed_data[processed_data.control_id == i] for i in [2, 3, 4]]
last_control_color_accuracy_array = [trial.color_accuracy.mean() for trial in last_control_trials]
last_control_identity_accuracy_array = [trial.identity_accuracy.mean() for trial in last_control_trials]
print(f"Performance on these two tasks remained constant on the final three control trials "
      f"(color: {last_control_color_accuracy_array[0] * 100}%, "
      f"{last_control_color_accuracy_array[1] * 100}%, and "
      f"{last_control_color_accuracy_array[2] * 100}% correct; "
      f"identity: {last_control_identity_accuracy_array[0] * 100}%, "
      f"{last_control_identity_accuracy_array[1] * 100}%, and "
      f"{last_control_identity_accuracy_array[2] * 100}% correct). ")

#%%
all_control_trials = [processed_data[processed_data.control_id == i] for i in [1, 2, 3, 4]]
all_control_location_accuracy_array = [trial.location_accuracy.mean() for trial in all_control_trials]
print(f"Participants’ performance on the location task was almost the "
      f"same on the surprise trial ({suprise_location_accuracy * 100}% correct) as on the control trials "
      f"({all_control_location_accuracy_array[0] * 100}%, {all_control_location_accuracy_array[1] * 100}%, "
      f"{all_control_location_accuracy_array[2] * 100}%, and {all_control_location_accuracy_array[3] * 100}% correct).")

#%%
import scipy.stats
import math

#%%
scipy.stats.chisquare([suprise_color_correct, first_control_color_correct],
)

#%%
chi2, p, dof, _ = scipy.stats.chi2_contingency(
    [
        [suprise_color_correct, first_control_color_correct],
        [suprise_color_total - suprise_color_correct, first_control_color_total - first_control_color_correct]
    ], correction=False
)
total = suprise_color_total + first_control_color_total
phi = math.sqrt(chi2 / total)

print(f"color: {first_control_color_accuracy * 100}% versus {suprise_color_accuracy * 100}%, "
      f"χ2({dof}, N = {total}) = {chi2}, p = {p}, ϕ = {phi};")

#%%
chi2, p, dof, _ = scipy.stats.chi2_contingency(
    [
        [suprise_identity_correct, first_control_identity_correct],
        [suprise_identity_total - suprise_identity_correct,
         first_control_identity_total - first_control_identity_correct]
    ], correction=False
)
total = suprise_identity_total + first_control_identity_total
phi = math.sqrt(chi2 / total)

print(f"identity: {first_control_identity_accuracy * 100}% versus {suprise_identity_accuracy * 100}%, "
      f"χ2({dof}, N = {total}) = {chi2}, p = {p}, ϕ = {phi};")
