#!/bin/bash
# Live functional tests FT-12 to FT-16, 2 October 2026, against the real services.
#
# Why a script: each test is the ordinary command (python main.py ...) with the
# answers to the two reviews piped in, so the commands can be repeated exactly
# and need nobody at the keyboard. What each command prints is also written to
# evidence/ft<NN>_*_20261002.log (for FT-12 to FT-15 the banner and the exit
# status go to the terminal only; a run that ends with "Run complete - brief
# written." returned 0). The decisions and requests go to output/run.log, as
# for every other run.
#
# Running it again overwrites those logs, and the results would differ: the
# replies of 2 October are now in the cache, and the services change.
#
# Why these five: FT-01 to FT-11 all used one research question, and the fixes
# of 1 October (remediation #12-#21) had only been unit-tested and replayed
# offline. FT-12 runs the revised plan-revision prompt (#18) for real; FT-13 to
# FT-15 are three other questions, chosen for variety (another field; another
# field with a constraint, #17; a vague question); FT-16 shows how a run that
# cannot start, or loses its input, now ends (#12, #21).
#
# Run from the project folder with the virtual environment active:
#     bash evidence/live_tests_20261002.sh            (all five; 21 minutes on 2 October)
#     bash evidence/live_tests_20261002.sh 14         (one test again)

cd "$(dirname "$0")/.." || exit 1
DAY=20261002

Q_STANDARD="What approaches reduce hallucination in LLM-based autonomous agents?"
Q_HEALTH="How effective is telemedicine for managing type 2 diabetes in adults?"
Q_SUPPLY="What interventions reduce food waste in retail supply chains?"
Q_VAGUE="is remote work good"

banner() { printf '\n=================== %s | %s ===================\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S')"; }

ft12() {  # Review 1: revise, with the reviewed plan now in the prompt, then approve twice
    banner "FT-12 revise the plan (standard question)"
    printf '%s\n' "merge sub-questions 3 and 4 into one about evaluation and keep three sub-questions in total" approve approve \
        | python main.py "$Q_STANDARD" 2>&1 | tee "evidence/ft12_revise_with_plan_${DAY}.log"
    echo "exit status: ${PIPESTATUS[1]}"
}

ft13() {  # a second research question, from another field
    banner "FT-13 second question (health)"
    printf '%s\n' approve approve | python main.py "$Q_HEALTH" 2>&1 | tee "evidence/ft13_second_question_${DAY}.log"
    echo "exit status: ${PIPESTATUS[1]}"
}

ft14() {  # a third question, with a constraint given on the command line
    banner "FT-14 third question with --constraints (supply chain)"
    printf '%s\n' approve approve | python main.py "$Q_SUPPLY" --constraints "empirical studies only, not reviews" 2>&1 \
        | tee "evidence/ft14_third_question_constraints_${DAY}.log"
    echo "exit status: ${PIPESTATUS[1]}"
}

ft15() {  # a vague, badly posed question
    banner "FT-15 vague question"
    printf '%s\n' approve approve | python main.py "$Q_VAGUE" 2>&1 | tee "evidence/ft15_vague_question_${DAY}.log"
    echo "exit status: ${PIPESTATUS[1]}"
}

ft16() {  # how a run ends when it cannot start, and when its input ends
    {
        banner "FT-16a no API key (.env set aside for this one command)"
        if [ -f .env ]; then
            trap '[ -f .env.off ] && mv .env.off .env' EXIT      # put the keys back whatever happens
            mv .env .env.off
            python main.py "test"; echo "exit status: $?"
            mv .env.off .env
            trap - EXIT
        else
            echo "skipped: no .env file here"
        fi
        banner "FT-16b input ends at the question prompt (as with Ctrl-D)"
        python main.py < /dev/null; echo "exit status: $?"
    } 2>&1 | tee "evidence/ft16_stop_messages_${DAY}.log"
}

for n in ${@:-12 13 14 15 16}; do
    case "$n" in
        12) ft12 ;; 13) ft13 ;; 14) ft14 ;; 15) ft15 ;; 16) ft16 ;;
        *) echo "unknown test: $n (use 12, 13, 14, 15 or 16)" ;;
    esac
done
banner "finished"
