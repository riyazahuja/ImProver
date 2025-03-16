theorem mem_symmDiff : a ∈ s ∆ t ↔ a ∈ s ∧ a ∉ t ∨ a ∈ t ∧ a ∉ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    a : α
    ⊢ Iff (Membership.mem (symmDiff s t) a) (Or (And (Membership.mem s a) (Not (Me …
  -/
  simp_rw [symmDiff, sup_eq_union, mem_union, mem_sdiff]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_symmDiff : (↑(s ∆ t) : Set α) = (s : Set α) ∆ t :=
                      /-
                        α : Type u_1
                        inst✝ : DecidableEq α
                        s t : Finset α
                        x : α
                        ⊢ Iff (Membership.mem (↑(symmDiff s t)) x) (Membership.mem (symmDiff ↑s ↑t) x)
                      -/
  Set.ext fun x => by simp [mem_symmDiff, Set.mem_symmDiff]
                      /-
                        🎉 no goals
                      -/


@[simp] lemma symmDiff_eq_empty : s ∆ t = ∅ ↔ s = t := symmDiff_eq_bot

@[simp] lemma symmDiff_nonempty : (s ∆ t).Nonempty ↔ s ≠ t :=
  nonempty_iff_ne_empty.trans symmDiff_eq_empty.not


