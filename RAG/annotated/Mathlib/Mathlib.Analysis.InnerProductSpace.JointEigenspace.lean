/-- The joint eigenspaces of a pair of symmetric operators form an
`OrthogonalFamily`. -/
theorem orthogonalFamily_eigenspace_inf_eigenspace (hA : A.IsSymmetric) (hB : B.IsSymmetric) :
    OrthogonalFamily 𝕜 (fun (i : 𝕜 × 𝕜) => (eigenspace A i.2 ⊓ eigenspace B i.1 : Submodule 𝕜 E))
      fun i => (eigenspace A i.2 ⊓ eigenspace B i.1).subtypeₗᵢ :=
  OrthogonalFamily.of_pairwise fun i j hij v ⟨hv1 , hv2⟩ ↦ by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      A B : LinearMap (RingHom.id 𝕜) E E
      hA : A.IsSymmetric
      hB : B.IsSymmetric
      i j : Prod 𝕜 𝕜
      hij : Ne i j
      v : E
      x✝ : Membership.mem (Min.min (Module.End.eigenspace A i.2) (Module.End.eigensp …
      hv1 : Membership.mem (↑(Module.End.eigenspace A i.2)) v
      hv2 : Membership.mem (↑(Module.End.eigenspace B i.1)) v
      ⊢ Membership.mem (Min.min (Module.End.eigenspace A j.2) (Module.End.eigenspace …
    -/
    obtain (h₁ | h₂) : i.1 ≠ j.1 ∨ i.2 ≠ j.2 := by rwa [Ne.eq_def, Prod.ext_iff, not_and_or] at hij
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      A B : LinearMap (RingHom.id 𝕜) E E
      hA : A.IsSymmetric
      hB : B.IsSymmetric
      i j : Prod 𝕜 𝕜
      hij : Ne i j
      v : E
      x✝ : Membership.mem (Min.min (Module.End.eigenspace A i.2) (Module.End.eigensp …
      hv1 : Membership.mem (↑(Module.End.eigenspace A i.2)) v
      hv2 : Membership.mem (↑(Module.End.eigenspace B i.1)) v
      h₁ : Ne i.1 j.1
      ⊢ Membership.mem (Min.min (Module.End.eigenspace A j.2) (Module.End.eigenspace …
    -/
    all_goals intro w ⟨hw1, hw2⟩
      /-
        case inl
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : RCLike 𝕜
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        A B : LinearMap (RingHom.id 𝕜) E E
        hA : A.IsSymmetric
        hB : B.IsSymmetric
        i j : Prod 𝕜 𝕜
        hij : Ne i j
        v : E
        x✝ : Membership.mem (Min.min (Module.End.eigenspace A i.2) (Module.End.eigensp …
        hv1 : Membership.mem (↑(Module.End.eigenspace A i.2)) v
        hv2 : Membership.mem (↑(Module.End.eigenspace B i.1)) v
        h₁ : Ne i.1 j.1
        w : E
        hw1 : Membership.mem (↑(Module.End.eigenspace A j.2)) w
        hw2 : Membership.mem (↑(Module.End.eigenspace B j.1)) w
        ⊢ Eq (Inner.inner w v) 0
      -/
    · exact hB.orthogonalFamily_eigenspaces.pairwise h₁ hv2 w hw2
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : RCLike 𝕜
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        A B : LinearMap (RingHom.id 𝕜) E E
        hA : A.IsSymmetric
        hB : B.IsSymmetric
        i j : Prod 𝕜 𝕜
        hij : Ne i j
        v : E
        x✝ : Membership.mem (Min.min (Module.End.eigenspace A i.2) (Module.End.eigensp …
        hv1 : Membership.mem (↑(Module.End.eigenspace A i.2)) v
        hv2 : Membership.mem (↑(Module.End.eigenspace B i.1)) v
        h₂ : Ne i.2 j.2
        w : E
        hw1 : Membership.mem (↑(Module.End.eigenspace A j.2)) w
        hw2 : Membership.mem (↑(Module.End.eigenspace B j.1)) w
        ⊢ Eq (Inner.inner w v) 0
      -/
    · exact hA.orthogonalFamily_eigenspaces.pairwise h₂ hv1 w hw1
      /-
        🎉 no goals
      -/


/-- The joint eigenspaces of a family of symmetric operators form an
`OrthogonalFamily`. -/
theorem orthogonalFamily_iInf_eigenspaces (hT : ∀ i, (T i).IsSymmetric) :
    OrthogonalFamily 𝕜 (fun γ : n → 𝕜 ↦ (⨅ j, eigenspace (T j) (γ j) : Submodule 𝕜 E))
      fun γ : n → 𝕜 ↦ (⨅ j, eigenspace (T j) (γ j)).subtypeₗᵢ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : n → Module.End 𝕜 E
    hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
    ⊢ OrthogonalFamily 𝕜 (fun γ => Subtype fun x => Membership.mem (iInf fun j =>  …
  -/
  intro f g hfg Ef Eg
  /-
    𝕜 : Type u_1
    E : Type u_2
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : n → Module.End 𝕜 E
    hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
    f g : n → 𝕜
    hfg : Ne f g
    Ef : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (f j)) x
    Eg : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (g j)) x
    ⊢ Eq (Inner.inner (((fun γ => (iInf fun j => (T j).eigenspace (γ j)).subtypeₗᵢ …
  -/
  obtain ⟨a , ha⟩ := Function.ne_iff.mp hfg
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : n → Module.End 𝕜 E
    hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
    f g : n → 𝕜
    hfg : Ne f g
    Ef : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (f j)) x
    Eg : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (g j)) x
    a : n
    ha : Ne (f a) (g a)
    ⊢ Eq (Inner.inner (((fun γ => (iInf fun j => (T j).eigenspace (γ j)).subtypeₗᵢ …
  -/
  have H := orthogonalFamily_eigenspaces (hT a) ha
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : n → Module.End 𝕜 E
    hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
    f g : n → 𝕜
    hfg : Ne f g
    Ef : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (f j)) x
    Eg : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (g j)) x
    a : n
    ha : Ne (f a) (g a)
    H : (fun i j => ∀ (v : (fun μ => Subtype fun x => Membership.mem ((T a).eigens …
    ⊢ Eq (Inner.inner (((fun γ => (iInf fun j => (T j).eigenspace (γ j)).subtypeₗᵢ …
  -/
  simp only [Submodule.coe_subtypeₗᵢ, Submodule.coe_subtype, Subtype.forall] at H
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    n : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : n → Module.End 𝕜 E
    hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
    f g : n → 𝕜
    hfg : Ne f g
    Ef : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (f j)) x
    Eg : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (g j)) x
    a : n
    ha : Ne (f a) (g a)
    H : ∀ (a_1 : E), Membership.mem ((T a).eigenspace (f a)) a_1 → ∀ (a_2 : E), Me …
    ⊢ Eq (Inner.inner (((fun γ => (iInf fun j => (T j).eigenspace (γ j)).subtypeₗᵢ …
  -/
  apply H
    /-
      case intro.b
      𝕜 : Type u_1
      E : Type u_2
      n : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : n → Module.End 𝕜 E
      hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
      f g : n → 𝕜
      hfg : Ne f g
      Ef : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (f j)) x
      Eg : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (g j)) x
      a : n
      ha : Ne (f a) (g a)
      H : ∀ (a_1 : E), Membership.mem ((T a).eigenspace (f a)) a_1 → ∀ (a_2 : E), Me …
      ⊢ Membership.mem ((T a).eigenspace (f a)) (((fun γ => (iInf fun j => (T j).eig …
    -/
  · exact (Submodule.mem_iInf <| fun _ ↦ eigenspace (T _) (f _)).mp Ef.2 _
    /-
      🎉 no goals
    -/
    /-
      case intro.b
      𝕜 : Type u_1
      E : Type u_2
      n : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : n → Module.End 𝕜 E
      hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
      f g : n → 𝕜
      hfg : Ne f g
      Ef : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (f j)) x
      Eg : Subtype fun x => Membership.mem (iInf fun j => (T j).eigenspace (g j)) x
      a : n
      ha : Ne (f a) (g a)
      H : ∀ (a_1 : E), Membership.mem ((T a).eigenspace (f a)) a_1 → ∀ (a_2 : E), Me …
      ⊢ Membership.mem ((T a).eigenspace (g a)) (((fun γ => (iInf fun j => (T j).eig …
    -/
  · exact (Submodule.mem_iInf <| fun _ ↦ eigenspace (T _) (g _)).mp Eg.2 _
    /-
      🎉 no goals
    -/


/-- If A and B are commuting symmetric operators on a finite dimensional inner product space
then the eigenspaces of the restriction of B to any eigenspace of A exhaust that eigenspace. -/
theorem iSup_eigenspace_inf_eigenspace_of_commute (hB : B.IsSymmetric) (hAB : Commute A B) :
    (⨆ γ, eigenspace A α ⊓ eigenspace B γ) = eigenspace A α := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    α : 𝕜
    A B : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hB : B.IsSymmetric
    hAB : Commute A B
    ⊢ Eq (iSup fun γ => Min.min (Module.End.eigenspace A α) (Module.End.eigenspace …
  -/
  conv_rhs => rw [← (eigenspace A α).map_subtype_top]
  simp only [← genEigenspace_eq_eigenspace (f := B), ← Submodule.map_iSup,
    (eigenspace A α).inf_genEigenspace _ (mapsTo_genEigenspace_of_comm hAB α 1)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    α : 𝕜
    A B : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hB : B.IsSymmetric
    hAB : Commute A B
    ⊢ Eq (Submodule.map (Module.End.eigenspace A α).subtype (iSup fun i => (Module …
  -/
  congr 1
  simpa only [genEigenspace_eq_eigenspace, Submodule.orthogonal_eq_bot_iff]
    using orthogonalComplement_iSup_eigenspaces_eq_bot <|
      hB.restrict_invariant <| mapsTo_genEigenspace_of_comm hAB α 1


/-- If A and B are commuting symmetric operators acting on a finite dimensional inner product space,
then the simultaneous eigenspaces of A and B exhaust the space. -/
theorem iSup_iSup_eigenspace_inf_eigenspace_eq_top_of_commute (hA : A.IsSymmetric)
    (hB : B.IsSymmetric) (hAB : Commute A B) :
    (⨆ α, ⨆ γ, eigenspace A α ⊓ eigenspace B γ) = ⊤ := by
  simpa [iSup_eigenspace_inf_eigenspace_of_commute hB hAB] using
    Submodule.orthogonal_eq_bot_iff.mp <| hA.orthogonalComplement_iSup_eigenspaces_eq_bot


/-- Given a commuting pair of symmetric linear operators on a finite dimensional inner product
space, the space decomposes as an internal direct sum of simultaneous eigenspaces of these
operators. -/
theorem directSum_isInternal_of_commute (hA : A.IsSymmetric) (hB : B.IsSymmetric)
    (hAB : Commute A B) :
    DirectSum.IsInternal (fun (i : 𝕜 × 𝕜) ↦ (eigenspace A i.2 ⊓ eigenspace B i.1)):= by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    A B : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hA : A.IsSymmetric
    hB : B.IsSymmetric
    hAB : Commute A B
    ⊢ DirectSum.IsInternal fun i => Min.min (Module.End.eigenspace A i.2) (Module. …
  -/
  apply (orthogonalFamily_eigenspace_inf_eigenspace hA hB).isInternal_iff.mpr
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    A B : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hA : A.IsSymmetric
    hB : B.IsSymmetric
    hAB : Commute A B
    ⊢ Eq (iSup fun i => Min.min (Module.End.eigenspace A i.2) (Module.End.eigenspa …
  -/
  rw [Submodule.orthogonal_eq_bot_iff, iSup_prod, iSup_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    A B : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hA : A.IsSymmetric
    hB : B.IsSymmetric
    hAB : Commute A B
    ⊢ Eq (iSup fun j => iSup fun i => Min.min (Module.End.eigenspace A { fst := i, …
  -/
  exact iSup_iSup_eigenspace_inf_eigenspace_eq_top_of_commute hA hB hAB
  /-
    🎉 no goals
  -/


/-- A commuting family of symmetric linear maps on a finite dimensional inner
product space is simultaneously diagonalizable. -/
theorem iSup_iInf_eq_top_of_commute {ι : Type*} {T : ι → E →ₗ[𝕜] E}
    (hT : ∀ i, (T i).IsSymmetric) (h : Pairwise (Commute on T)):
    ⨆ χ : ι → 𝕜, ⨅ i, eigenspace (T i) (χ i) = ⊤ :=
  calc
  _ = ⨆ χ : ι → 𝕜, ⨅ i, maxGenEigenspace (T i) (χ i) :=
    congr(⨆ χ : ι → 𝕜, ⨅ i,
      $(maxGenEigenspace_eq_eigenspace (isFinitelySemisimple <| hT _) (χ _))).symm
  _ = ⊤ :=
    iSup_iInf_maxGenEigenspace_eq_top_of_iSup_maxGenEigenspace_eq_top_of_commute T h fun _ ↦ by
    rw [← orthogonal_eq_bot_iff,
      congr(⨆ μ, $(maxGenEigenspace_eq_eigenspace (isFinitelySemisimple <| hT _) μ)),
      (hT _).orthogonalComplement_iSup_eigenspaces_eq_bot]


/-- In finite dimensions, given a commuting family of symmetric linear operators, the inner
product space on which they act decomposes as an internal direct sum of joint eigenspaces. -/
theorem LinearMap.IsSymmetric.directSum_isInternal_of_pairwise_commute [DecidableEq (n → 𝕜)]
    (hT : ∀ i, (T i).IsSymmetric) (hC : Pairwise (Commute on T)) :
    DirectSum.IsInternal (fun α : n → 𝕜 ↦ ⨅ j, eigenspace (T j) (α j)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    n : Type u_3
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    T : n → Module.End 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : DecidableEq (n → 𝕜)
    hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
    hC : Pairwise (Function.onFun Commute T)
    ⊢ DirectSum.IsInternal fun α => iInf fun j => (T j).eigenspace (α j)
  -/
  rw [OrthogonalFamily.isInternal_iff]
    /-
      𝕜 : Type u_1
      E : Type u_2
      n : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      T : n → Module.End 𝕜 E
      inst✝¹ : FiniteDimensional 𝕜 E
      inst✝ : DecidableEq (n → 𝕜)
      hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
      hC : Pairwise (Function.onFun Commute T)
      ⊢ Eq (iSup fun α => iInf fun j => (T j).eigenspace (α j)).orthogonal Bot.bot
    -/
  · rw [iSup_iInf_eq_top_of_commute hT hC, top_orthogonal_eq_bot]
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      n : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      T : n → Module.End 𝕜 E
      inst✝¹ : FiniteDimensional 𝕜 E
      inst✝ : DecidableEq (n → 𝕜)
      hT : ∀ (i : n), LinearMap.IsSymmetric (T i)
      hC : Pairwise (Function.onFun Commute T)
      ⊢ OrthogonalFamily 𝕜 (fun i => Subtype fun x => Membership.mem (iInf fun j =>  …
    -/
  · exact orthogonalFamily_iInf_eigenspaces hT
    /-
      🎉 no goals
    -/


