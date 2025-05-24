#include <stdio.h>
// #include <conio.h> // Optional

#define MAX_P 10 // Max processes
#define MAX_R 10 // Max resources

int main() {
    // clrscr(); // Optional
    int P, R; // Number of processes and resources
    int i, j, k;

    int total_instances[MAX_R];        // Total instances of each resource
    int allocation[MAX_P][MAX_R];
    int max[MAX_P][MAX_R];
    int available[MAX_R];              // Calculated available
    int need[MAX_P][MAX_R];

    int finish[MAX_P];
    int safeSeq[MAX_P];
    int work[MAX_R];
    int count = 0;
    int pass_found_process;

    printf("Enter the number of processes (max %d): ", MAX_P);
    scanf("%d", &P);
    if (P > MAX_P || P <= 0) {
        printf("Invalid number of processes.\n"); return 1;
    }

    printf("Enter the number of resource types (max %d): ", MAX_R);
    scanf("%d", &R);
    if (R > MAX_R || R <= 0) {
        printf("Invalid number of resource types.\n"); return 1;
    }

    printf("\nEnter the number of instances for each resource type:\n");
    for (j = 0; j < R; j++) {
        printf("Instances of resource R%d: ", j + 1);
        scanf("%d", &total_instances[j]);
        available[j] = total_instances[j]; // Initialize available with total
    }

    printf("\nEnter the Allocation Matrix (%dx%d):\n", P, R);
    for (i = 0; i < P; i++) {
        printf("P%d: ", i); // Changed from P%d: to P%d to match P0, P1 style if desired
        for (j = 0; j < R; j++) {
            scanf("%d", &allocation[i][j]);
            available[j] -= allocation[i][j]; // Subtract allocated from total to get current available
        }
    }

    printf("\nEnter the Max Matrix (%dx%d):\n", P, R);
    for (i = 0; i < P; i++) {
        printf("P%d: ", i);
        for (j = 0; j < R; j++) {
            scanf("%d", &max[i][j]);
        }
    }

    printf("\n--- Process Details & Need Matrix ---\n");
    for (i = 0; i < P; i++) {
        printf("Process P%d:\n", i);
        for (j = 0; j < R; j++) {
            need[i][j] = max[i][j] - allocation[i][j];
            printf("  R%d: allocated %d, maximum %d, need %d\n",
                   j + 1, allocation[i][j], max[i][j], need[i][j]);
            // Optional: Add the validation for negative need here if you want the program to stop
            // if (need[i][j] < 0) {
            //     printf("Error: P%d has Allocation > Max for R%d. Banker's cannot proceed meaningfully.\n", i, j+1);
            //     // getch();
            //     return 1;
            // }
        }
    }

    printf("\nInitial Availability Vector:\n");
    for (j = 0; j < R; j++) {
        printf("R%d: %d  ", j + 1, available[j]);
    }
    printf("\n");

    // --- Safety Algorithm ---
    for (i = 0; i < P; i++) {
        finish[i] = 0;
    }
    for (j = 0; j < R; j++) {
        work[j] = available[j];
    }
    count = 0;

    printf("\n--- Safe Sequence Check ---\n");
    for (k = 0; k < P; k++) {
        pass_found_process = 0;
        for (i = 0; i < P; i++) {
            if (finish[i] == 0) {
                int can_allocate = 1;
                for (j = 0; j < R; j++) {
                    if (need[i][j] > work[j]) {
                        can_allocate = 0;
                        break;
                    }
                }

                if (can_allocate == 1) {
                    printf("P%d can be allocated. Work becomes: ", i);
                    for (j = 0; j < R; j++) {
                        work[j] += allocation[i][j];
                        printf("R%d: %d  ", j + 1, work[j]);
                    }
                    printf("\n");
                    safeSeq[count++] = i;
                    finish[i] = 1;
                    pass_found_process = 1;
                }
            }
        }
        if (pass_found_process == 0 && count < P) {
            // No process found in this pass, and not all are finished.
        }
    }
    // --- End Safety Algorithm ---

    if (count == P) {
        printf("\nSystem is in a SAFE state.\nSafe sequence is: ");
        for (i = 0; i < P; i++) {
            printf("P%d ", safeSeq[i]);
	}
	printf("\n");
    } else {
	printf("\nSystem is in an UNSAFE state.\n");
	printf("%d out of %d processes could be safely allocated.\n", count, P);
    }

    getch();
    return 0;
}