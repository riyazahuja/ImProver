/-- Left-multiplication by a nonzero element of a topological division ring is proper, i.e.,
inverse images of compact sets are compact. -/
theorem Filter.tendsto_cocompact_mul_left₀ [ContinuousMul K] {a : K} (ha : a ≠ 0) :
    Filter.Tendsto (fun x : K => a * x) (Filter.cocompact K) (Filter.cocompact K) :=
  Filter.tendsto_cocompact_mul_left (inv_mul_cancel₀ ha)


/-- Right-multiplication by a nonzero element of a topological division ring is proper, i.e.,
inverse images of compact sets are compact. -/
theorem Filter.tendsto_cocompact_mul_right₀ [ContinuousMul K] {a : K} (ha : a ≠ 0) :
    Filter.Tendsto (fun x : K => x * a) (Filter.cocompact K) (Filter.cocompact K) :=
  Filter.tendsto_cocompact_mul_right (mul_inv_cancel₀ ha)


/-- Compact hausdorff topological fields are finite. -/
instance (priority := 100) {K} [DivisionRing K] [TopologicalSpace K]
    [TopologicalRing K] [CompactSpace K] [T2Space K] : Finite K := by
  suffices DiscreteTopology K by
    exact finite_of_compact_of_discrete
  /-
    K✝ : Type u_1
    inst✝⁶ : DivisionRing K✝
    inst✝⁵ : TopologicalSpace K✝
    K : Type u_2
    inst✝⁴ : DivisionRing K
    inst✝³ : TopologicalSpace K
    inst✝² : TopologicalRing K
    inst✝¹ : CompactSpace K
    inst✝ : T2Space K
    ⊢ DiscreteTopology K
  -/
  rw [discreteTopology_iff_isOpen_singleton_zero]
  /-
    K✝ : Type u_1
    inst✝⁶ : DivisionRing K✝
    inst✝⁵ : TopologicalSpace K✝
    K : Type u_2
    inst✝⁴ : DivisionRing K
    inst✝³ : TopologicalSpace K
    inst✝² : TopologicalRing K
    inst✝¹ : CompactSpace K
    inst✝ : T2Space K
    ⊢ IsOpen (Singleton.singleton 0)
  -/
  exact GroupWithZero.isOpen_singleton_zero
  /-
    🎉 no goals
  -/


/-- A topological division ring is a division ring with a topology where all operations are
    continuous, including inversion. -/
class TopologicalDivisionRing extends TopologicalRing K, HasContinuousInv₀ K : Prop


/-- The (topological-space) closure of a subfield of a topological field is
itself a subfield. -/
def Subfield.topologicalClosure (K : Subfield α) : Subfield α :=
  { K.toSubring.topologicalClosure with
    carrier := _root_.closure (K : Set α)
    inv_mem' := fun x hx => by
      /-
        K✝ : Type u_1
        inst✝⁴ : DivisionRing K✝
        inst✝³ : TopologicalSpace K✝
        α : Type u_2
        inst✝² : Field α
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalDivisionRing α
        K : Subfield α
        x : α
        hx : Membership.mem { carrier := _root_.closure ↑K, mul_mem' := ⋯, one_mem' := …
        ⊢ Membership.mem { carrier := _root_.closure ↑K, mul_mem' := ⋯, one_mem' := ⋯, …
      -/
      dsimp only at hx ⊢
      /-
        K✝ : Type u_1
        inst✝⁴ : DivisionRing K✝
        inst✝³ : TopologicalSpace K✝
        α : Type u_2
        inst✝² : Field α
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalDivisionRing α
        K : Subfield α
        x : α
        hx : Membership.mem (_root_.closure ↑K) x
        ⊢ Membership.mem (_root_.closure ↑K) (Inv.inv x)
      -/
      rcases eq_or_ne x 0 with (rfl | h)
        /-
          case inl
          K✝ : Type u_1
          inst✝⁴ : DivisionRing K✝
          inst✝³ : TopologicalSpace K✝
          α : Type u_2
          inst✝² : Field α
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalDivisionRing α
          K : Subfield α
          hx : Membership.mem (_root_.closure ↑K) 0
          ⊢ Membership.mem (_root_.closure ↑K) (Inv.inv 0)
        -/
      · rwa [inv_zero]
        /-
          🎉 no goals
        -/
      · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: Lean fails to find InvMemClass instance
        /-
          case inr
          K✝ : Type u_1
          inst✝⁴ : DivisionRing K✝
          inst✝³ : TopologicalSpace K✝
          α : Type u_2
          inst✝² : Field α
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalDivisionRing α
          K : Subfield α
          x : α
          hx : Membership.mem (_root_.closure ↑K) x
          h : Ne x 0
          ⊢ Membership.mem (_root_.closure ↑K) (Inv.inv x)
        -/
        rw [← @inv_coe_set α (Subfield α) _ _ SubfieldClass.toInvMemClass K, ← Set.image_inv_eq_inv]
        /-
          case inr
          K✝ : Type u_1
          inst✝⁴ : DivisionRing K✝
          inst✝³ : TopologicalSpace K✝
          α : Type u_2
          inst✝² : Field α
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalDivisionRing α
          K : Subfield α
          x : α
          hx : Membership.mem (_root_.closure ↑K) x
          h : Ne x 0
          ⊢ Membership.mem (_root_.closure (Set.image (fun x => Inv.inv x) ↑K)) (Inv.inv …
        -/
        exact mem_closure_image (continuousAt_inv₀ h) hx }
        /-
          🎉 no goals
        -/


theorem Subfield.le_topologicalClosure (s : Subfield α) : s ≤ s.topologicalClosure :=
  _root_.subset_closure


theorem Subfield.isClosed_topologicalClosure (s : Subfield α) :
    IsClosed (s.topologicalClosure : Set α) :=
  isClosed_closure


theorem Subfield.topologicalClosure_minimal (s : Subfield α) {t : Subfield α} (h : s ≤ t)
    (ht : IsClosed (t : Set α)) : s.topologicalClosure ≤ t :=
  closure_minimal h ht


/--
The map `fun x => a * x + b`, as a homeomorphism from `𝕜` (a topological field) to itself,
when `a ≠ 0`.
-/
@[simps]
def affineHomeomorph (a b : 𝕜) (h : a ≠ 0) : 𝕜 ≃ₜ 𝕜 where
  toFun x := a * x + b
  invFun y := (y - b) / a
  left_inv x := by
    /-
      K : Type u_1
      inst✝⁴ : DivisionRing K
      inst✝³ : TopologicalSpace K
      𝕜 : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : TopologicalRing 𝕜
      a b : 𝕜
      h : Ne a 0
      x : 𝕜
      ⊢ Eq ((fun y => HDiv.hDiv (HSub.hSub y b) a) ((fun x => HAdd.hAdd (HMul.hMul a …
    -/
    simp only [add_sub_cancel_right]
    /-
      K : Type u_1
      inst✝⁴ : DivisionRing K
      inst✝³ : TopologicalSpace K
      𝕜 : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : TopologicalRing 𝕜
      a b : 𝕜
      h : Ne a 0
      x : 𝕜
      ⊢ Eq (HDiv.hDiv (HMul.hMul a x) a) x
    -/
    exact mul_div_cancel_left₀ x h
    /-
      🎉 no goals
    -/
                    /-
                      K : Type u_1
                      inst✝⁴ : DivisionRing K
                      inst✝³ : TopologicalSpace K
                      𝕜 : Type u_2
                      inst✝² : Field 𝕜
                      inst✝¹ : TopologicalSpace 𝕜
                      inst✝ : TopologicalRing 𝕜
                      a b : 𝕜
                      h : Ne a 0
                      y : 𝕜
                      ⊢ Eq ((fun x => HAdd.hAdd (HMul.hMul a x) b) ((fun y => HDiv.hDiv (HSub.hSub y …
                    -/
  right_inv y := by simp [mul_div_cancel₀ _ h]
                    /-
                      🎉 no goals
                    -/


theorem affineHomeomorph_image_Icc {𝕜 : Type*} [LinearOrderedField 𝕜] [TopologicalSpace 𝕜]
    [TopologicalRing 𝕜] (a b c d : 𝕜) (h : 0 < a) :
    affineHomeomorph a b h.ne' '' Set.Icc c d = Set.Icc (a * c + b) (a * d + b) := by
  /-
    𝕜 : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b c d : 𝕜
    h : LT.lt 0 a
    ⊢ Eq (Set.image (⇑(affineHomeomorph a b ⋯)) (Set.Icc c d)) (Set.Icc (HAdd.hAdd …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem affineHomeomorph_image_Ico {𝕜 : Type*} [LinearOrderedField 𝕜] [TopologicalSpace 𝕜]
    [TopologicalRing 𝕜] (a b c d : 𝕜) (h : 0 < a) :
    affineHomeomorph a b h.ne' '' Set.Ico c d = Set.Ico (a * c + b) (a * d + b) := by
  /-
    𝕜 : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b c d : 𝕜
    h : LT.lt 0 a
    ⊢ Eq (Set.image (⇑(affineHomeomorph a b ⋯)) (Set.Ico c d)) (Set.Ico (HAdd.hAdd …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem affineHomeomorph_image_Ioc {𝕜 : Type*} [LinearOrderedField 𝕜] [TopologicalSpace 𝕜]
    [TopologicalRing 𝕜] (a b c d : 𝕜) (h : 0 < a) :
    affineHomeomorph a b h.ne' '' Set.Ioc c d = Set.Ioc (a * c + b) (a * d + b) := by
  /-
    𝕜 : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b c d : 𝕜
    h : LT.lt 0 a
    ⊢ Eq (Set.image (⇑(affineHomeomorph a b ⋯)) (Set.Ioc c d)) (Set.Ioc (HAdd.hAdd …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem affineHomeomorph_image_Ioo {𝕜 : Type*} [LinearOrderedField 𝕜] [TopologicalSpace 𝕜]
    [TopologicalRing 𝕜] (a b c d : 𝕜) (h : 0 < a) :
    affineHomeomorph a b h.ne' '' Set.Ioo c d = Set.Ioo (a * c + b) (a * d + b) := by
  /-
    𝕜 : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : TopologicalRing 𝕜
    a b c d : 𝕜
    h : LT.lt 0 a
    ⊢ Eq (Set.image (⇑(affineHomeomorph a b ⋯)) (Set.Ioo c d)) (Set.Ioo (HAdd.hAdd …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem IsLocalMin.inv {f : α → β} {a : α} (h1 : IsLocalMin f a) (h2 : ∀ᶠ z in 𝓝 a, 0 < f z) :
    IsLocalMax f⁻¹ a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrderedSemifield β
    f : α → β
    a : α
    h1 : IsLocalMin f a
    h2 : Filter.Eventually (fun z => LT.lt 0 (f z)) (nhds a)
    ⊢ IsLocalMax (Inv.inv f) a
  -/
  filter_upwards [h1, h2] with z h3 h4 using(inv_le_inv₀ h4 h2.self_of_nhds).mpr h3
  /-
    🎉 no goals
  -/


/-- If `f` is a function `α → 𝕜` which is continuous on a preconnected set `S`, and
`f ^ 2 = 1` on `S`, then either `f = 1` on `S`, or `f = -1` on `S`. -/
theorem IsPreconnected.eq_one_or_eq_neg_one_of_sq_eq [Ring 𝕜] [NoZeroDivisors 𝕜]
    (hS : IsPreconnected S) (hf : ContinuousOn f S) (hsq : EqOn (f ^ 2) 1 S) :
    EqOn f 1 S ∨ EqOn f (-1) S := by
  /-
    α : Type u_2
    𝕜 : Type u_3
    f : α → 𝕜
    S : Set α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace 𝕜
    inst✝² : T1Space 𝕜
    inst✝¹ : Ring 𝕜
    inst✝ : NoZeroDivisors 𝕜
    hS : IsPreconnected S
    hf : ContinuousOn f S
    hsq : Set.EqOn (HPow.hPow f 2) 1 S
    ⊢ Or (Set.EqOn f 1 S) (Set.EqOn f (-1) S)
  -/
  have : DiscreteTopology ({1, -1} : Set 𝕜) := Finite.instDiscreteTopology
  have hmaps : MapsTo f S {1, -1} := by
    simpa only [EqOn, Pi.one_apply, Pi.pow_apply, sq_eq_one_iff] using hsq
  /-
    α : Type u_2
    𝕜 : Type u_3
    f : α → 𝕜
    S : Set α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : TopologicalSpace 𝕜
    inst✝² : T1Space 𝕜
    inst✝¹ : Ring 𝕜
    inst✝ : NoZeroDivisors 𝕜
    hS : IsPreconnected S
    hf : ContinuousOn f S
    hsq : Set.EqOn (HPow.hPow f 2) 1 S
    this : DiscreteTopology ↑(Insert.insert 1 (Singleton.singleton (-1)))
    hmaps : Set.MapsTo f S (Insert.insert 1 (Singleton.singleton (-1)))
    ⊢ Or (Set.EqOn f 1 S) (Set.EqOn f (-1) S)
  -/
  simpa using hS.eqOn_const_of_mapsTo hf hmaps
  /-
    🎉 no goals
  -/


/-- If `f, g` are functions `α → 𝕜`, both continuous on a preconnected set `S`, with
`f ^ 2 = g ^ 2` on `S`, and `g z ≠ 0` all `z ∈ S`, then either `f = g` or `f = -g` on
`S`. -/
theorem IsPreconnected.eq_or_eq_neg_of_sq_eq [Field 𝕜] [HasContinuousInv₀ 𝕜] [ContinuousMul 𝕜]
    (hS : IsPreconnected S) (hf : ContinuousOn f S) (hg : ContinuousOn g S)
    (hsq : EqOn (f ^ 2) (g ^ 2) S) (hg_ne : ∀ {x : α}, x ∈ S → g x ≠ 0) :
    EqOn f g S ∨ EqOn f (-g) S := by
  have hsq : EqOn ((f / g) ^ 2) 1 S := fun x hx => by
    simpa [div_eq_one_iff_eq (pow_ne_zero _ (hg_ne hx)), div_pow] using hsq hx
  simpa (config := { contextual := true }) [EqOn, div_eq_iff (hg_ne _)]
    using hS.eq_one_or_eq_neg_one_of_sq_eq (hf.div hg fun z => hg_ne) hsq


/-- If `f, g` are functions `α → 𝕜`, both continuous on a preconnected set `S`, with
`f ^ 2 = g ^ 2` on `S`, and `g z ≠ 0` all `z ∈ S`, then as soon as `f = g` holds at
one point of `S` it holds for all points. -/
theorem IsPreconnected.eq_of_sq_eq [Field 𝕜] [HasContinuousInv₀ 𝕜] [ContinuousMul 𝕜]
    (hS : IsPreconnected S) (hf : ContinuousOn f S) (hg : ContinuousOn g S)
    (hsq : EqOn (f ^ 2) (g ^ 2) S) (hg_ne : ∀ {x : α}, x ∈ S → g x ≠ 0) {y : α} (hy : y ∈ S)
    (hy' : f y = g y) : EqOn f g S := fun x hx => by
  /-
    α : Type u_2
    𝕜 : Type u_3
    f g : α → 𝕜
    S : Set α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : T1Space 𝕜
    inst✝² : Field 𝕜
    inst✝¹ : HasContinuousInv₀ 𝕜
    inst✝ : ContinuousMul 𝕜
    hS : IsPreconnected S
    hf : ContinuousOn f S
    hg : ContinuousOn g S
    hsq : Set.EqOn (HPow.hPow f 2) (HPow.hPow g 2) S
    hg_ne : ∀ {x : α}, Membership.mem S x → Ne (g x) 0
    y : α
    hy : Membership.mem S y
    hy' : Eq (f y) (g y)
    x : α
    hx : Membership.mem S x
    ⊢ Eq (f x) (g x)
  -/
  rcases hS.eq_or_eq_neg_of_sq_eq hf hg @hsq @hg_ne with (h | h)
    /-
      case inl
      α : Type u_2
      𝕜 : Type u_3
      f g : α → 𝕜
      S : Set α
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : T1Space 𝕜
      inst✝² : Field 𝕜
      inst✝¹ : HasContinuousInv₀ 𝕜
      inst✝ : ContinuousMul 𝕜
      hS : IsPreconnected S
      hf : ContinuousOn f S
      hg : ContinuousOn g S
      hsq : Set.EqOn (HPow.hPow f 2) (HPow.hPow g 2) S
      hg_ne : ∀ {x : α}, Membership.mem S x → Ne (g x) 0
      y : α
      hy : Membership.mem S y
      hy' : Eq (f y) (g y)
      x : α
      hx : Membership.mem S x
      h : Set.EqOn f g S
      ⊢ Eq (f x) (g x)
    -/
  · exact h hx
    /-
      🎉 no goals
    -/
  · rw [h _, Pi.neg_apply, neg_eq_iff_add_eq_zero, ← two_mul, mul_eq_zero,
                                                          /-
                                                            case inr
                                                            α : Type u_2
                                                            𝕜 : Type u_3
                                                            f g : α → 𝕜
                                                            S : Set α
                                                            inst✝⁵ : TopologicalSpace α
                                                            inst✝⁴ : TopologicalSpace 𝕜
                                                            inst✝³ : T1Space 𝕜
                                                            inst✝² : Field 𝕜
                                                            inst✝¹ : HasContinuousInv₀ 𝕜
                                                            inst✝ : ContinuousMul 𝕜
                                                            hS : IsPreconnected S
                                                            hf : ContinuousOn f S
                                                            hg : ContinuousOn g S
                                                            hsq : Set.EqOn (HPow.hPow f 2) (HPow.hPow g 2) S
                                                            hg_ne : ∀ {x : α}, Membership.mem S x → Ne (g x) 0
                                                            y : α
                                                            hy : Membership.mem S y
                                                            hy' : Or (Eq 2 0) False
                                                            x : α
                                                            hx : Membership.mem S x
                                                            h : Set.EqOn f (Neg.neg g) S
                                                            ⊢ Or (Eq 2 0) False
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
      (iff_of_eq (iff_false _)).2 (hg_ne _)] at hy' ⊢ <;> assumption
                                                          /-
                                                            🎉 no goals
                                                          -/


