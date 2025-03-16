@[ext]
lemma ext_inner {A B : E →WOT[𝕜] F} (h : ∀ x y, ⟪y, A x⟫_𝕜 = ⟪y, B x⟫_𝕜) : A = B := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : TopologicalSpace E
    inst✝² : Module 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace 𝕜 F
    A B : ContinuousLinearMapWOT 𝕜 E F
    h : ∀ (x : E) (y : F), Eq (Inner.inner y (A x)) (Inner.inner y (B x))
    ⊢ Eq A B
  -/
  rw [ext_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : TopologicalSpace E
    inst✝² : Module 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace 𝕜 F
    A B : ContinuousLinearMapWOT 𝕜 E F
    h : ∀ (x : E) (y : F), Eq (Inner.inner y (A x)) (Inner.inner y (B x))
    ⊢ ∀ (x : E), Eq (A x) (B x)
  -/
  exact fun x => ext_inner_left 𝕜 fun y => h x y
  /-
    🎉 no goals
  -/


open Filter in
/-- The defining property of the weak operator topology: a function `f` tends to
`A : E →WOT[𝕜] F` along filter `l` iff `⟪y, (f a) x⟫` tends to `⟪y, A x⟫` along the same filter. -/
lemma tendsto_iff_forall_inner_apply_tendsto [CompleteSpace F] {α : Type*} {l : Filter α}
    {f : α → E →WOT[𝕜] F} {A : E →WOT[𝕜] F} :
    Tendsto f l (𝓝 A) ↔ ∀ x y, Tendsto (fun a => ⟪y, (f a) x⟫_𝕜) l (𝓝 ⟪y, A x⟫_𝕜) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    α : Type u_4
    l : Filter α
    f : α → ContinuousLinearMapWOT 𝕜 E F
    A : ContinuousLinearMapWOT 𝕜 E F
    ⊢ Iff (Filter.Tendsto f l (nhds A)) (∀ (x : E) (y : F), Filter.Tendsto (fun a  …
  -/
  simp_rw [tendsto_iff_forall_dual_apply_tendsto, ← InnerProductSpace.toDual_apply]
  exact .symm <| forall_congr' fun _ ↦
    Equiv.forall_congr (InnerProductSpace.toDual 𝕜 F) fun _ ↦ Iff.rfl


lemma le_nhds_iff_forall_inner_apply_le_nhds [CompleteSpace F] {l : Filter (E →WOT[𝕜] F)}
    {A : E →WOT[𝕜] F} : l ≤ 𝓝 A ↔ ∀ x y, l.map (fun T => ⟪y, T x⟫_𝕜) ≤ 𝓝 (⟪y, A x⟫_𝕜) :=
  tendsto_iff_forall_inner_apply_tendsto (f := id)


