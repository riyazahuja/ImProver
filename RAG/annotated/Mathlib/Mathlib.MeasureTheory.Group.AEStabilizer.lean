/-- A.e. stabilizer of a set under a group action. -/
@[to_additive (attr := simps) "A.e. stabilizer of a set under an additive group action."]
def aestabilizer (s : Set α) : Subgroup G where
  carrier := {g | g • s =ᵐ[μ] s}
                 /-
                   G : Type u_1
                   α : Type u_2
                   inst✝² : Group G
                   inst✝¹ : MulAction G α
                   x✝ : MeasurableSpace α
                   μ : MeasureTheory.Measure α
                   inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                   s : Set α
                   ⊢ Membership.mem { carrier := setOf fun g => (MeasureTheory.ae μ).EventuallyEq …
                 -/
  one_mem' := by simp
                               /-
                                 G : Type u_1
                                 α : Type u_2
                                 inst✝² : Group G
                                 inst✝¹ : MulAction G α
                                 x✝ : MeasurableSpace α
                                 μ : MeasureTheory.Measure α
                                 inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                                 s : Set α
                                 g₁ g₂ : G
                                 h₁ : Membership.mem (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (HSMul.h …
                                 h₂ : Membership.mem (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (HSMul.h …
                                 ⊢ Membership.mem (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMu …
                               -/
                 /-
                   🎉 no goals
                 -/
                               /-
                                 🎉 no goals
                               -/
  -- TODO: `calc` would be more readable but fails because of defeq abuse
  mul_mem' {g₁ g₂} h₁ h₂ := by simpa only [smul_smul] using ((smul_set_ae_eq g₁).2 h₂).trans h₁
                       /-
                         G : Type u_1
                         α : Type u_2
                         inst✝² : Group G
                         inst✝¹ : MulAction G α
                         x✝ : MeasurableSpace α
                         μ : MeasureTheory.Measure α
                         inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                         s : Set α
                         g : G
                         h : Membership.mem { carrier := setOf fun g => (MeasureTheory.ae μ).Eventually …
                         ⊢ Membership.mem { carrier := setOf fun g => (MeasureTheory.ae μ).EventuallyEq …
                       -/
  inv_mem' {g} h := by simpa using (smul_set_ae_eq g⁻¹).2 h.out.symm
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp)]
lemma mem_aestabilizer : g ∈ aestabilizer G μ s ↔ g • s =ᵐ[μ] s := .rfl


@[to_additive]
lemma stabilizer_le_aestabilizer (s : Set α) : stabilizer G s ≤ aestabilizer G μ s := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s : Set α
    ⊢ LE.le (MulAction.stabilizer G s) (MulAction.aestabilizer G μ s)
  -/
  intro g hg
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s : Set α
    g : G
    hg : Membership.mem (MulAction.stabilizer G s) g
    ⊢ Membership.mem (MulAction.aestabilizer G μ s) g
  -/
  simp_all
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                                             /-
                                                                               G : Type u_1
                                                                               α : Type u_2
                                                                               inst✝² : Group G
                                                                               inst✝¹ : MulAction G α
                                                                               x✝² : MeasurableSpace α
                                                                               μ : MeasureTheory.Measure α
                                                                               inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                                                                               x✝¹ : G
                                                                               x✝ : Membership.mem Top.top x✝¹
                                                                               ⊢ Membership.mem (MulAction.aestabilizer G μ EmptyCollection.emptyCollection)  …
                                                                             -/
lemma aestabilizer_empty : aestabilizer G μ ∅ = ⊤ := top_unique fun _ _ ↦ by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[to_additive (attr := simp)]
                                                                               /-
                                                                                 G : Type u_1
                                                                                 α : Type u_2
                                                                                 inst✝² : Group G
                                                                                 inst✝¹ : MulAction G α
                                                                                 x✝² : MeasurableSpace α
                                                                                 μ : MeasureTheory.Measure α
                                                                                 inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                                                                                 x✝¹ : G
                                                                                 x✝ : Membership.mem Top.top x✝¹
                                                                                 ⊢ Membership.mem (MulAction.aestabilizer G μ Set.univ) x✝¹
                                                                               -/
lemma aestabilizer_univ : aestabilizer G μ univ = ⊤ := top_unique fun _ _ ↦ by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[to_additive]
lemma aestabilizer_congr (h : s =ᵐ[μ] t) : aestabilizer G μ s = aestabilizer G μ t := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Eq (MulAction.aestabilizer G μ s) (MulAction.aestabilizer G μ t)
  -/
  ext g
  /-
    case h
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s t
    g : G
    ⊢ Iff (Membership.mem (MulAction.aestabilizer G μ s) g) (Membership.mem (MulAc …
  -/
  rw [mem_aestabilizer, mem_aestabilizer, h.congr_right, ((smul_set_ae_eq g).2 h).congr_left]
  /-
    🎉 no goals
  -/


lemma aestabilizer_of_aeconst (hs : EventuallyConst s (ae μ)) : aestabilizer G μ s = ⊤ := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    s : Set α
    hs : Filter.EventuallyConst s (MeasureTheory.ae μ)
    ⊢ Eq (MulAction.aestabilizer G μ s) Top.top
  -/
  refine top_unique fun g _ ↦ ?_
  cases eventuallyConst_set'.mp hs with
  | inl h => simp [aestabilizer_congr h]
  | inr h => simp [aestabilizer_congr h]


@[to_additive]
theorem smul_ae_eq_self_of_mem_zpowers (hs : (x • s : Set α) =ᵐ[μ] s)
    (hy : y ∈ Subgroup.zpowers x) : (y • s : Set α) =ᵐ[μ] s := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    x y : G
    s : Set α
    hs : (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul x s) s
    hy : Membership.mem (Subgroup.zpowers x) y
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul y s) s
  -/
  rw [← MulAction.mem_aestabilizer, ← Subgroup.zpowers_le] at hs
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    x y : G
    s : Set α
    hs : LE.le (Subgroup.zpowers x) (MulAction.aestabilizer G μ s)
    hy : Membership.mem (Subgroup.zpowers x) y
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul y s) s
  -/
  exact hs hy
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inv_smul_ae_eq_self (hs : (x • s : Set α) =ᵐ[μ] s) : (x⁻¹ • s : Set α) =ᵐ[μ] s :=
  inv_mem (s := MulAction.aestabilizer G μ s) hs


