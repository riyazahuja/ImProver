/-- Least common multiple of a multiset -/
def lcm (s : Multiset α) : α :=
  s.fold GCDMonoid.lcm 1


@[simp]
theorem lcm_zero : (0 : Multiset α).lcm = 1 :=
  fold_zero _ _


@[simp]
theorem lcm_cons (a : α) (s : Multiset α) : (a ::ₘ s).lcm = GCDMonoid.lcm a s.lcm :=
  fold_cons_left _ _ _ _


@[simp]
theorem lcm_singleton {a : α} : ({a} : Multiset α).lcm = normalize a :=
  (fold_singleton _ _ _).trans <| lcm_one_right _


@[simp]
theorem lcm_add (s₁ s₂ : Multiset α) : (s₁ + s₂).lcm = GCDMonoid.lcm s₁.lcm s₂.lcm :=
               /-
                 α : Type u_1
                 inst✝¹ : CancelCommMonoidWithZero α
                 inst✝ : NormalizedGCDMonoid α
                 s₁ s₂ : Multiset α
                 ⊢ Eq (HAdd.hAdd s₁ s₂).lcm (Multiset.fold GCDMonoid.lcm (GCDMonoid.lcm 1 1) (H …
               -/
  Eq.trans (by simp [lcm]) (fold_add _ _ _ _ _)
               /-
                 🎉 no goals
               -/


theorem lcm_dvd {s : Multiset α} {a : α} : s.lcm ∣ a ↔ ∀ b ∈ s, b ∣ a :=
                              /-
                                α : Type u_1
                                inst✝¹ : CancelCommMonoidWithZero α
                                inst✝ : NormalizedGCDMonoid α
                                s : Multiset α
                                a : α
                                ⊢ Iff (Dvd.dvd (Multiset.lcm 0) a) (∀ (b : α), Membership.mem 0 b → Dvd.dvd b a)
                              -/
  Multiset.induction_on s (by simp)
                              /-
                                🎉 no goals
                              -/
        /-
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : NormalizedGCDMonoid α
          s : Multiset α
          a : α
          ⊢ ∀ (a_1 : α) (s : Multiset α), Iff (Dvd.dvd s.lcm a) (∀ (b : α), Membership.m …
        -/
    (by simp +contextual [or_imp, forall_and, lcm_dvd_iff])
        /-
          🎉 no goals
        -/


theorem dvd_lcm {s : Multiset α} {a : α} (h : a ∈ s) : a ∣ s.lcm :=
  lcm_dvd.1 dvd_rfl _ h


theorem lcm_mono {s₁ s₂ : Multiset α} (h : s₁ ⊆ s₂) : s₁.lcm ∣ s₂.lcm :=
  lcm_dvd.2 fun _ hb ↦ dvd_lcm (h hb)

/- Porting note: Following `Algebra.GCDMonoid.Basic`'s version of `normalize_gcd`, I'm giving
this lower priority to avoid linter complaints about simp-normal form -/
/- Porting note: Mathport seems to be replacing `Multiset.induction_on s $` with
`(Multiset.induction_on s)`, when it should be `Multiset.induction_on s <|`. -/

@[simp 1100]
theorem normalize_lcm (s : Multiset α) : normalize s.lcm = s.lcm :=
                              /-
                                α : Type u_1
                                inst✝¹ : CancelCommMonoidWithZero α
                                inst✝ : NormalizedGCDMonoid α
                                s : Multiset α
                                ⊢ Eq (normalize (Multiset.lcm 0)) (Multiset.lcm 0)
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) fun a s _ ↦ by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
nonrec theorem lcm_eq_zero_iff [Nontrivial α] (s : Multiset α) : s.lcm = 0 ↔ (0 : α) ∈ s := by
  induction s using Multiset.induction_on with
  | empty => simp only [lcm_zero, one_ne_zero, not_mem_zero]
  | cons a s ihs => simp only [mem_cons, lcm_cons, lcm_eq_zero_iff, ihs, @eq_comm _ a]


@[simp]
theorem lcm_dedup (s : Multiset α) : (dedup s).lcm = s.lcm :=
                              /-
                                α : Type u_1
                                inst✝² : CancelCommMonoidWithZero α
                                inst✝¹ : NormalizedGCDMonoid α
                                inst✝ : DecidableEq α
                                s : Multiset α
                                ⊢ Eq (Multiset.dedup 0).lcm (Multiset.lcm 0)
                              -/
  Multiset.induction_on s (by simp) fun a s IH ↦ by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.lcm s.lcm
      ⊢ Eq (Multiset.cons a s).dedup.lcm (Multiset.cons a s).lcm
    -/
    by_cases h : a ∈ s <;> simp [IH, h]
                           /-
                             🎉 no goals
                           -/
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.lcm s.lcm
      h : Membership.mem s a
      ⊢ Eq s.lcm (GCDMonoid.lcm a s.lcm)
    -/
    unfold lcm
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.lcm s.lcm
      h : Membership.mem s a
      ⊢ Eq (Multiset.fold GCDMonoid.lcm 1 s) (GCDMonoid.lcm a (Multiset.fold GCDMono …
    -/
    rw [← cons_erase h, fold_cons_left, ← lcm_assoc, lcm_same]
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.lcm s.lcm
      h : Membership.mem s a
      ⊢ Eq (GCDMonoid.lcm a (Multiset.fold GCDMonoid.lcm 1 (s.erase a))) (GCDMonoid. …
    -/
    apply lcm_eq_of_associated_left (associated_normalize _)
    /-
      🎉 no goals
    -/


@[simp]
theorem lcm_ndunion (s₁ s₂ : Multiset α) : (ndunion s₁ s₂).lcm = GCDMonoid.lcm s₁.lcm s₂.lcm := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (s₁.ndunion s₂).lcm (GCDMonoid.lcm s₁.lcm s₂.lcm)
  -/
  rw [← lcm_dedup, dedup_ext.2, lcm_dedup, lcm_add]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ ∀ (a : α), Iff (Membership.mem (s₁.ndunion s₂) a) (Membership.mem (HAdd.hAdd …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_union (s₁ s₂ : Multiset α) : (s₁ ∪ s₂).lcm = GCDMonoid.lcm s₁.lcm s₂.lcm := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (Union.union s₁ s₂).lcm (GCDMonoid.lcm s₁.lcm s₂.lcm)
  -/
  rw [← lcm_dedup, dedup_ext.2, lcm_dedup, lcm_add]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ ∀ (a : α), Iff (Membership.mem (Union.union s₁ s₂) a) (Membership.mem (HAdd. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_ndinsert (a : α) (s : Multiset α) : (ndinsert a s).lcm = GCDMonoid.lcm a s.lcm := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.ndinsert a s).lcm (GCDMonoid.lcm a s.lcm)
  -/
  rw [← lcm_dedup, dedup_ext.2, lcm_dedup, lcm_cons]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ ∀ (a_1 : α), Iff (Membership.mem (Multiset.ndinsert a s) a_1) (Membership.me …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Greatest common divisor of a multiset -/
def gcd (s : Multiset α) : α :=
  s.fold GCDMonoid.gcd 0


@[simp]
theorem gcd_zero : (0 : Multiset α).gcd = 0 :=
  fold_zero _ _


@[simp]
theorem gcd_cons (a : α) (s : Multiset α) : (a ::ₘ s).gcd = GCDMonoid.gcd a s.gcd :=
  fold_cons_left _ _ _ _


@[simp]
theorem gcd_singleton {a : α} : ({a} : Multiset α).gcd = normalize a :=
  (fold_singleton _ _ _).trans <| gcd_zero_right _


@[simp]
theorem gcd_add (s₁ s₂ : Multiset α) : (s₁ + s₂).gcd = GCDMonoid.gcd s₁.gcd s₂.gcd :=
               /-
                 α : Type u_1
                 inst✝¹ : CancelCommMonoidWithZero α
                 inst✝ : NormalizedGCDMonoid α
                 s₁ s₂ : Multiset α
                 ⊢ Eq (HAdd.hAdd s₁ s₂).gcd (Multiset.fold GCDMonoid.gcd (GCDMonoid.gcd 0 0) (H …
               -/
  Eq.trans (by simp [gcd]) (fold_add _ _ _ _ _)
               /-
                 🎉 no goals
               -/


theorem dvd_gcd {s : Multiset α} {a : α} : a ∣ s.gcd ↔ ∀ b ∈ s, a ∣ b :=
                              /-
                                α : Type u_1
                                inst✝¹ : CancelCommMonoidWithZero α
                                inst✝ : NormalizedGCDMonoid α
                                s : Multiset α
                                a : α
                                ⊢ Iff (Dvd.dvd a (Multiset.gcd 0)) (∀ (b : α), Membership.mem 0 b → Dvd.dvd a b)
                              -/
  Multiset.induction_on s (by simp)
                              /-
                                🎉 no goals
                              -/
        /-
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : NormalizedGCDMonoid α
          s : Multiset α
          a : α
          ⊢ ∀ (a_1 : α) (s : Multiset α), Iff (Dvd.dvd a s.gcd) (∀ (b : α), Membership.m …
        -/
    (by simp +contextual [or_imp, forall_and, dvd_gcd_iff])
        /-
          🎉 no goals
        -/


theorem gcd_dvd {s : Multiset α} {a : α} (h : a ∈ s) : s.gcd ∣ a :=
  dvd_gcd.1 dvd_rfl _ h


theorem gcd_mono {s₁ s₂ : Multiset α} (h : s₁ ⊆ s₂) : s₂.gcd ∣ s₁.gcd :=
  dvd_gcd.2 fun _ hb ↦ gcd_dvd (h hb)

/- Porting note: Following `Algebra.GCDMonoid.Basic`'s version of `normalize_gcd`, I'm giving
this lower priority to avoid linter complaints about simp-normal form -/

@[simp 1100]
theorem normalize_gcd (s : Multiset α) : normalize s.gcd = s.gcd :=
                              /-
                                α : Type u_1
                                inst✝¹ : CancelCommMonoidWithZero α
                                inst✝ : NormalizedGCDMonoid α
                                s : Multiset α
                                ⊢ Eq (normalize (Multiset.gcd 0)) (Multiset.gcd 0)
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) fun a s _ ↦ by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem gcd_eq_zero_iff (s : Multiset α) : s.gcd = 0 ↔ ∀ x : α, x ∈ s → x = 0 := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Multiset α
    ⊢ Iff (Eq s.gcd 0) (∀ (x : α), Membership.mem s x → Eq x 0)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Multiset α
      ⊢ Eq s.gcd 0 → ∀ (x : α), Membership.mem s x → Eq x 0
    -/
  · intro h x hx
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Multiset α
      h : Eq s.gcd 0
      x : α
      hx : Membership.mem s x
      ⊢ Eq x 0
    -/
    apply eq_zero_of_zero_dvd
    /-
      case mp.h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Multiset α
      h : Eq s.gcd 0
      x : α
      hx : Membership.mem s x
      ⊢ Dvd.dvd 0 x
    -/
    rw [← h]
    /-
      case mp.h
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Multiset α
      h : Eq s.gcd 0
      x : α
      hx : Membership.mem s x
      ⊢ Dvd.dvd s.gcd x
    -/
    apply gcd_dvd hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Multiset α
      ⊢ (∀ (x : α), Membership.mem s x → Eq x 0) → Eq s.gcd 0
    -/
  · refine s.induction_on ?_ ?_
      /-
        case mpr.refine_1
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : NormalizedGCDMonoid α
        s : Multiset α
        ⊢ (∀ (x : α), Membership.mem 0 x → Eq x 0) → Eq (Multiset.gcd 0) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case mpr.refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Multiset α
      ⊢ ∀ (a : α) (s : Multiset α), ((∀ (x : α), Membership.mem s x → Eq x 0) → Eq s …
    -/
    intro a s sgcd h
    /-
      case mpr.refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s✝ : Multiset α
      a : α
      s : Multiset α
      sgcd : (∀ (x : α), Membership.mem s x → Eq x 0) → Eq s.gcd 0
      h : ∀ (x : α), Membership.mem (Multiset.cons a s) x → Eq x 0
      ⊢ Eq (Multiset.cons a s).gcd 0
    -/
    simp [h a (mem_cons_self a s), sgcd fun x hx ↦ h x (mem_cons_of_mem hx)]
    /-
      🎉 no goals
    -/


theorem gcd_map_mul (a : α) (s : Multiset α) : (s.map (a * ·)).gcd = normalize a * s.gcd := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.map (fun x => HMul.hMul a x) s).gcd (HMul.hMul (normalize a) s. …
  -/
  refine s.induction_on ?_ fun b s ih ↦ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      a : α
      s : Multiset α
      ⊢ Eq (Multiset.map (fun x => HMul.hMul a x) 0).gcd (HMul.hMul (normalize a) (M …
    -/
  · simp_rw [map_zero, gcd_zero, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      a : α
      s✝ : Multiset α
      b : α
      s : Multiset α
      ih : Eq (Multiset.map (fun x => HMul.hMul a x) s).gcd (HMul.hMul (normalize a) …
      ⊢ Eq (Multiset.map (fun x => HMul.hMul a x) (Multiset.cons b s)).gcd (HMul.hMu …
    -/
  · simp_rw [map_cons, gcd_cons, ← gcd_mul_left]
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      a : α
      s✝ : Multiset α
      b : α
      s : Multiset α
      ih : Eq (Multiset.map (fun x => HMul.hMul a x) s).gcd (HMul.hMul (normalize a) …
      ⊢ Eq (GCDMonoid.gcd (HMul.hMul a b) (Multiset.map (fun x => HMul.hMul a x) s). …
    -/
    rw [ih]
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      a : α
      s✝ : Multiset α
      b : α
      s : Multiset α
      ih : Eq (Multiset.map (fun x => HMul.hMul a x) s).gcd (HMul.hMul (normalize a) …
      ⊢ Eq (GCDMonoid.gcd (HMul.hMul a b) (HMul.hMul (normalize a) s.gcd)) (GCDMonoi …
    -/
    apply ((normalize_associated a).mul_right _).gcd_eq_right
    /-
      🎉 no goals
    -/


@[simp]
theorem gcd_dedup (s : Multiset α) : (dedup s).gcd = s.gcd :=
                              /-
                                α : Type u_1
                                inst✝² : CancelCommMonoidWithZero α
                                inst✝¹ : NormalizedGCDMonoid α
                                inst✝ : DecidableEq α
                                s : Multiset α
                                ⊢ Eq (Multiset.dedup 0).gcd (Multiset.gcd 0)
                              -/
  Multiset.induction_on s (by simp) fun a s IH ↦ by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.gcd s.gcd
      ⊢ Eq (Multiset.cons a s).dedup.gcd (Multiset.cons a s).gcd
    -/
    by_cases h : a ∈ s <;> simp [IH, h]
                           /-
                             🎉 no goals
                           -/
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.gcd s.gcd
      h : Membership.mem s a
      ⊢ Eq s.gcd (GCDMonoid.gcd a s.gcd)
    -/
    unfold gcd
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.gcd s.gcd
      h : Membership.mem s a
      ⊢ Eq (Multiset.fold GCDMonoid.gcd 0 s) (GCDMonoid.gcd a (Multiset.fold GCDMono …
    -/
    rw [← cons_erase h, fold_cons_left, ← gcd_assoc, gcd_same]
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : NormalizedGCDMonoid α
      inst✝ : DecidableEq α
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : Eq s.dedup.gcd s.gcd
      h : Membership.mem s a
      ⊢ Eq (GCDMonoid.gcd a (Multiset.fold GCDMonoid.gcd 0 (s.erase a))) (GCDMonoid. …
    -/
    apply (associated_normalize _).gcd_eq_left
    /-
      🎉 no goals
    -/


@[simp]
theorem gcd_ndunion (s₁ s₂ : Multiset α) : (ndunion s₁ s₂).gcd = GCDMonoid.gcd s₁.gcd s₂.gcd := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (s₁.ndunion s₂).gcd (GCDMonoid.gcd s₁.gcd s₂.gcd)
  -/
  rw [← gcd_dedup, dedup_ext.2, gcd_dedup, gcd_add]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ ∀ (a : α), Iff (Membership.mem (s₁.ndunion s₂) a) (Membership.mem (HAdd.hAdd …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_union (s₁ s₂ : Multiset α) : (s₁ ∪ s₂).gcd = GCDMonoid.gcd s₁.gcd s₂.gcd := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (Union.union s₁ s₂).gcd (GCDMonoid.gcd s₁.gcd s₂.gcd)
  -/
  rw [← gcd_dedup, dedup_ext.2, gcd_dedup, gcd_add]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ ∀ (a : α), Iff (Membership.mem (Union.union s₁ s₂) a) (Membership.mem (HAdd. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_ndinsert (a : α) (s : Multiset α) : (ndinsert a s).gcd = GCDMonoid.gcd a s.gcd := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.ndinsert a s).gcd (GCDMonoid.gcd a s.gcd)
  -/
  rw [← gcd_dedup, dedup_ext.2, gcd_dedup, gcd_cons]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ ∀ (a_1 : α), Iff (Membership.mem (Multiset.ndinsert a s) a_1) (Membership.me …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem extract_gcd' (s t : Multiset α) (hs : ∃ x, x ∈ s ∧ x ≠ (0 : α))
    (ht : s = t.map (s.gcd * ·)) : t.gcd = 1 :=
  ((@mul_right_eq_self₀ _ _ s.gcd _).1 <| by
        /-
          α : Type u_1
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : NormalizedGCDMonoid α
          s t : Multiset α
          hs : Exists fun x => And (Membership.mem s x) (Ne x 0)
          ht : Eq s (Multiset.map (fun x => HMul.hMul s.gcd x) t)
          ⊢ Eq (HMul.hMul s.gcd t.gcd) s.gcd
        -/
        conv_lhs => rw [← normalize_gcd, ← gcd_map_mul, ← ht]).resolve_right <| by
        /-
          🎉 no goals
        -/
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s t : Multiset α
      hs : Exists fun x => And (Membership.mem s x) (Ne x 0)
      ht : Eq s (Multiset.map (fun x => HMul.hMul s.gcd x) t)
      ⊢ Not (Eq s.gcd 0)
    -/
    contrapose! hs
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s t : Multiset α
      ht : Eq s (Multiset.map (fun x => HMul.hMul s.gcd x) t)
      hs : Eq s.gcd 0
      ⊢ ∀ (x : α), Membership.mem s x → Eq x 0
    -/
    exact s.gcd_eq_zero_iff.1 hs
    /-
      🎉 no goals
    -/

/- Porting note: The old proof used a strange form
`have := _, refine ⟨s.pmap @f (fun _ ↦ id), this, extract_gcd' s _ h this⟩,`
so I rearranged the proof slightly. -/

theorem extract_gcd (s : Multiset α) (hs : s ≠ 0) :
    ∃ t : Multiset α, s = t.map (s.gcd * ·) ∧ t.gcd = 1 := by
  classical
    by_cases h : ∀ x ∈ s, x = (0 : α)
    · use replicate (card s) 1
      rw [map_replicate, eq_replicate, mul_one, s.gcd_eq_zero_iff.2 h, ← nsmul_singleton,
    ← gcd_dedup, dedup_nsmul (card_pos.2 hs).ne', dedup_singleton, gcd_singleton]
      exact ⟨⟨rfl, h⟩, normalize_one⟩
    · choose f hf using @gcd_dvd _ _ _ s
      push_neg at h
      refine ⟨s.pmap @f fun _ ↦ id, ?_, extract_gcd' s _ h ?_⟩ <;>
      · rw [map_pmap]
        conv_lhs => rw [← s.map_id, ← s.pmap_eq_map _ _ fun _ ↦ id]
        congr with (x hx)
        rw [id, ← hf hx]


