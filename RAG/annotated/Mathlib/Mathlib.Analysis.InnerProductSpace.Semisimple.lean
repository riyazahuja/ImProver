/-- The orthogonal complement of an invariant submodule is invariant. -/
lemma orthogonalComplement_mem_invtSubmodule (hp : p ∈ T.invtSubmodule) :
    pᗮ ∈ T.invtSubmodule :=
  fun x hx y hy ↦ hT y x ▸ hx (T y) (hp hy)


/-- Symmetric operators are semisimple on finite-dimensional subspaces. -/
theorem isFinitelySemisimple :
    T.IsFinitelySemisimple := by
  refine Module.End.isFinitelySemisimple_iff.mpr fun p hp₁ hp₂ q hq₁ hq₂ ↦
    ⟨qᗮ ⊓ p, inf_le_right, Module.End.invtSubmodule.inf_mem ?_ hp₁, ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : Module.End 𝕜 E
      hT : LinearMap.IsSymmetric T
      p : Submodule 𝕜 E
      hp₁ : Membership.mem T.invtSubmodule p
      hp₂ : Module.Finite 𝕜 (Subtype fun x => Membership.mem p x)
      q : Submodule 𝕜 E
      hq₁ : Membership.mem T.invtSubmodule q
      hq₂ : LE.le q p
      ⊢ Membership.mem T.invtSubmodule q.orthogonal
    -/
  · exact orthogonalComplement_mem_invtSubmodule hT hq₁
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : Module.End 𝕜 E
      hT : LinearMap.IsSymmetric T
      p : Submodule 𝕜 E
      hp₁ : Membership.mem T.invtSubmodule p
      hp₂ : Module.Finite 𝕜 (Subtype fun x => Membership.mem p x)
      q : Submodule 𝕜 E
      hq₁ : Membership.mem T.invtSubmodule q
      hq₂ : LE.le q p
      ⊢ Disjoint q (Min.min q.orthogonal p)
    -/
  · simp [disjoint_iff, ← inf_assoc, Submodule.inf_orthogonal_eq_bot q]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : Module.End 𝕜 E
      hT : LinearMap.IsSymmetric T
      p : Submodule 𝕜 E
      hp₁ : Membership.mem T.invtSubmodule p
      hp₂ : Module.Finite 𝕜 (Subtype fun x => Membership.mem p x)
      q : Submodule 𝕜 E
      hq₁ : Membership.mem T.invtSubmodule q
      hq₂ : LE.le q p
      ⊢ Eq (Max.max q (Min.min q.orthogonal p)) p
    -/
  · suffices q ⊔ qᗮ = ⊤ by rw [← sup_inf_assoc_of_le _ hq₂, this, top_inf_eq p]
    /-
      case refine_3
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : Module.End 𝕜 E
      hT : LinearMap.IsSymmetric T
      p : Submodule 𝕜 E
      hp₁ : Membership.mem T.invtSubmodule p
      hp₂ : Module.Finite 𝕜 (Subtype fun x => Membership.mem p x)
      q : Submodule 𝕜 E
      hq₁ : Membership.mem T.invtSubmodule q
      hq₂ : LE.le q p
      ⊢ Eq (Max.max q q.orthogonal) Top.top
    -/
    replace hp₂ : Module.Finite 𝕜 q := Submodule.finiteDimensional_of_le hq₂
    /-
      case refine_3
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : Module.End 𝕜 E
      hT : LinearMap.IsSymmetric T
      p : Submodule 𝕜 E
      hp₁ : Membership.mem T.invtSubmodule p
      q : Submodule 𝕜 E
      hq₁ : Membership.mem T.invtSubmodule q
      hq₂ : LE.le q p
      hp₂ : Module.Finite 𝕜 (Subtype fun x => Membership.mem q x)
      ⊢ Eq (Max.max q q.orthogonal) Top.top
    -/
    exact Submodule.sup_orthogonal_of_completeSpace
    /-
      🎉 no goals
    -/


