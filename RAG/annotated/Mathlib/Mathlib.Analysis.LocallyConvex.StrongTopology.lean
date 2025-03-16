theorem locallyConvexSpace (𝔖 : Set (Set E)) (h𝔖₁ : 𝔖.Nonempty)
    (h𝔖₂ : DirectedOn (· ⊆ ·) 𝔖) :
    LocallyConvexSpace R (UniformConvergenceCLM σ F 𝔖) := by
  apply LocallyConvexSpace.ofBasisZero _ _ _ _
    (UniformConvergenceCLM.hasBasis_nhds_zero_of_basis _ _ _ h𝔖₁ h𝔖₂
      (LocallyConvexSpace.convex_basis_zero R F)) _
  /-
    R : Type u_1
    𝕜₁ : Type u_2
    𝕜₂ : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝¹³ : AddCommGroup E
    inst✝¹² : TopologicalSpace E
    inst✝¹¹ : AddCommGroup F
    inst✝¹⁰ : TopologicalSpace F
    inst✝⁹ : TopologicalAddGroup F
    inst✝⁸ : OrderedSemiring R
    inst✝⁷ : NormedField 𝕜₁
    inst✝⁶ : NormedField 𝕜₂
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : Module 𝕜₂ F
    σ : RingHom 𝕜₁ 𝕜₂
    inst✝³ : Module R F
    inst✝² : ContinuousConstSMul R F
    inst✝¹ : LocallyConvexSpace R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    h𝔖₁ : 𝔖.Nonempty
    h𝔖₂ : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) 𝔖
    ⊢ ∀ (i : Prod (Set E) (Set F)), And (Membership.mem 𝔖 i.1) (And (Membership.me …
  -/
  rintro ⟨S, V⟩ ⟨_, _, hVconvex⟩ f hf g hg a b ha hb hab x hx
  /-
    case mk.intro.intro
    R : Type u_1
    𝕜₁ : Type u_2
    𝕜₂ : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝¹³ : AddCommGroup E
    inst✝¹² : TopologicalSpace E
    inst✝¹¹ : AddCommGroup F
    inst✝¹⁰ : TopologicalSpace F
    inst✝⁹ : TopologicalAddGroup F
    inst✝⁸ : OrderedSemiring R
    inst✝⁷ : NormedField 𝕜₁
    inst✝⁶ : NormedField 𝕜₂
    inst✝⁵ : Module 𝕜₁ E
    inst✝⁴ : Module 𝕜₂ F
    σ : RingHom 𝕜₁ 𝕜₂
    inst✝³ : Module R F
    inst✝² : ContinuousConstSMul R F
    inst✝¹ : LocallyConvexSpace R F
    inst✝ : SMulCommClass 𝕜₂ R F
    𝔖 : Set (Set E)
    h𝔖₁ : 𝔖.Nonempty
    h𝔖₂ : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) 𝔖
    S : Set E
    V : Set F
    left✝¹ : Membership.mem 𝔖 { fst := S, snd := V }.1
    left✝ : Membership.mem (nhds 0) { fst := S, snd := V }.2
    hVconvex : Convex R { fst := S, snd := V }.2
    f : UniformConvergenceCLM σ F 𝔖
    hf : Membership.mem (setOf fun f => ∀ (x : E), Membership.mem { fst := S, snd  …
    g : UniformConvergenceCLM σ F 𝔖
    hg : Membership.mem (setOf fun f => ∀ (x : E), Membership.mem { fst := S, snd  …
    a b : R
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x : E
    hx : Membership.mem { fst := S, snd := V }.1 x
    ⊢ Membership.mem (id { fst := S, snd := V }.2) ((HAdd.hAdd (HSMul.hSMul a f) ( …
  -/
  exact hVconvex (hf x hx) (hg x hx) ha hb hab
  /-
    🎉 no goals
  -/


instance instLocallyConvexSpace : LocallyConvexSpace R (E →SL[σ] F) :=
  UniformConvergenceCLM.locallyConvexSpace R _ ⟨∅, Bornology.isVonNBounded_empty 𝕜₁ E⟩
    (directedOn_of_sup_mem fun _ _ => Bornology.IsVonNBounded.union)


