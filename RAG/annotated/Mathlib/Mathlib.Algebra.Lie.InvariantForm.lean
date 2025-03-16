variable (L) in
/--
A bilinear form on a Lie module `M` of a Lie algebra `L` is *invariant* if
for all `x : L` and `y z : M` the condition `Φ ⁅x, y⁆ z = -Φ y ⁅x, z⁆` holds.
-/
def _root_.LinearMap.BilinForm.lieInvariant : Prop :=
  ∀ (x : L) (y z : M), Φ ⁅x, y⁆ z = -Φ y ⁅x, z⁆


lemma _root_.LinearMap.BilinForm.lieInvariant_iff [LieAlgebra R L] [LieModule R L M] :
    Φ.lieInvariant L ↔ Φ ∈ LieModule.maxTrivSubmodule R L (LinearMap.BilinForm R M) := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    Φ : LinearMap.BilinForm R M
    inst✝¹ : LieAlgebra R L
    inst✝ : LieModule R L M
    ⊢ Iff (LinearMap.BilinForm.lieInvariant L Φ) (Membership.mem (LieModule.maxTri …
  -/
  refine ⟨fun h x ↦ ?_, fun h x y z ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      Φ : LinearMap.BilinForm R M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      h : LinearMap.BilinForm.lieInvariant L Φ
      x : L
      ⊢ Eq (Bracket.bracket x Φ) 0
    -/
  · ext y z
    rw [LieHom.lie_apply, LinearMap.sub_apply, Module.Dual.lie_apply, LinearMap.zero_apply,
      LinearMap.zero_apply, h, sub_self]
    /-
      case refine_2
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      Φ : LinearMap.BilinForm R M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      h : Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap.BilinForm R M)) Φ
      x : L
      y z : M
      ⊢ Eq ((Φ (Bracket.bracket x y)) z) (Neg.neg ((Φ y) (Bracket.bracket x z)))
    -/
  · replace h := LinearMap.congr_fun₂ (h x) y z
    simp only [LieHom.lie_apply, LinearMap.sub_apply, Module.Dual.lie_apply,
      LinearMap.zero_apply, sub_eq_zero] at h
    /-
      case refine_2
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      Φ : LinearMap.BilinForm R M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      x : L
      y z : M
      h : Eq (Neg.neg ((Φ y) (Bracket.bracket x z))) ((Φ (Bracket.bracket x y)) z)
      ⊢ Eq ((Φ (Bracket.bracket x y)) z) (Neg.neg ((Φ y) (Bracket.bracket x z)))
    -/
    simp [← h]
    /-
      🎉 no goals
    -/


/--
The orthogonal complement of a Lie submodule `N` with respect to an invariant bilinear form `Φ` is
the Lie submodule of elements `y` such that `Φ x y = 0` for all `x ∈ N`.
-/
@[simps!]
def orthogonal (hΦ_inv : Φ.lieInvariant L) (N : LieSubmodule R L M) : LieSubmodule R L M where
  __ := Φ.orthogonal N
  lie_mem {x y} := by
    suffices (∀ n ∈ N, Φ n y = 0) → ∀ n ∈ N, Φ n ⁅x, y⁆ = 0 by
      simpa only [LinearMap.BilinForm.isOrtho_def, -- and some default simp lemmas
        AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup, Submodule.mem_toAddSubmonoid,
        LinearMap.BilinForm.mem_orthogonal_iff, LieSubmodule.mem_toSubmodule]
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      Φ : LinearMap.BilinForm R M
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      N : LieSubmodule R L M
      x : L
      y : M
      ⊢ (∀ (n : M), Membership.mem N n → Eq ((Φ n) y) 0) → ∀ (n : M), Membership.mem …
    -/
    intro H a ha
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      Φ : LinearMap.BilinForm R M
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      N : LieSubmodule R L M
      x : L
      y : M
      H : ∀ (n : M), Membership.mem N n → Eq ((Φ n) y) 0
      a : M
      ha : Membership.mem N a
      ⊢ Eq ((Φ a) (Bracket.bracket x y)) 0
    -/
    rw [← neg_eq_zero, ← hΦ_inv]
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      Φ : LinearMap.BilinForm R M
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      N : LieSubmodule R L M
      x : L
      y : M
      H : ∀ (n : M), Membership.mem N n → Eq ((Φ n) y) 0
      a : M
      ha : Membership.mem N a
      ⊢ Eq ((Φ (Bracket.bracket x a)) y) 0
    -/
    exact H _ <| N.lie_mem ha
    /-
      🎉 no goals
    -/


@[simp]
lemma orthogonal_toSubmodule (N : LieSubmodule R L M) :
    (orthogonal Φ hΦ_inv N).toSubmodule = Φ.orthogonal N.toSubmodule := rfl


lemma mem_orthogonal (N : LieSubmodule R L M) (y : M) :
    y ∈ orthogonal Φ hΦ_inv N ↔ ∀ x ∈ N, Φ x y = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    Φ : LinearMap.BilinForm R M
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    N : LieSubmodule R L M
    y : M
    ⊢ Iff (Membership.mem (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv N) y) (∀ ( …
  -/
  simp [orthogonal, LinearMap.BilinForm.isOrtho_def, LinearMap.BilinForm.mem_orthogonal_iff]
  /-
    🎉 no goals
  -/


lemma orthogonal_disjoint
    (Φ : LinearMap.BilinForm R L) (hΦ_nondeg : Φ.Nondegenerate) (hΦ_inv : Φ.lieInvariant L)
    -- TODO: replace the following assumption by a typeclass assumption `[HasNonAbelianAtoms]`
    (hL : ∀ I : LieIdeal R L, IsAtom I → ¬IsLieAbelian I)
    (I : LieIdeal R L) (hI : IsAtom I) :
    Disjoint I (orthogonal Φ hΦ_inv I) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    ⊢ Disjoint I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
  -/
  rw [disjoint_iff, ← hI.lt_iff, lt_iff_le_and_ne]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    ⊢ And (LE.le (Min.min I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)) I) ( …
  -/
  suffices ¬I ≤ orthogonal Φ hΦ_inv I by simpa
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    ⊢ Not (LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I))
  -/
  intro contra
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    ⊢ False
  -/
  apply hI.1
  rw [eq_bot_iff, ← lie_eq_self_of_isAtom_of_nonabelian I hI (hL I hI),
      LieSubmodule.lieIdeal_oper_eq_span, LieSubmodule.lieSpan_le]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket …
  -/
  rintro _ ⟨x, y, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    x y : Subtype fun x => Membership.mem I x
    ⊢ Membership.mem (↑Bot.bot) (Bracket.bracket ↑x ↑y)
  -/
  simp only [LieSubmodule.bot_coe, Set.mem_singleton_iff]
  /-
    case intro.intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    x y : Subtype fun x => Membership.mem I x
    ⊢ Eq (Bracket.bracket ↑x ↑y) 0
  -/
  apply hΦ_nondeg
  /-
    case intro.intro.a
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    x y : Subtype fun x => Membership.mem I x
    ⊢ ∀ (n : L), Eq ((Φ (Bracket.bracket ↑x ↑y)) n) 0
  -/
  intro z
  /-
    case intro.intro.a
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    x y : Subtype fun x => Membership.mem I x
    z : L
    ⊢ Eq ((Φ (Bracket.bracket ↑x ↑y)) z) 0
  -/
  rw [hΦ_inv, neg_eq_zero]
  /-
    case intro.intro.a
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    x y : Subtype fun x => Membership.mem I x
    z : L
    ⊢ Eq ((Φ ↑y) (Bracket.bracket (↑x) z)) 0
  -/
  have hyz : ⁅(x : L), z⁆ ∈ I := lie_mem_left _ _ _ _ _ x.2
  /-
    case intro.intro.a
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    Φ : LinearMap.BilinForm R L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hL : ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal R L
    hI : IsAtom I
    contra : LE.le I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
    x y : Subtype fun x => Membership.mem I x
    z : L
    hyz : Membership.mem I (Bracket.bracket (↑x) z)
    ⊢ Eq ((Φ ↑y) (Bracket.bracket (↑x) z)) 0
  -/
  exact contra hyz y y.2
  /-
    🎉 no goals
  -/


open Module Submodule in
lemma orthogonal_isCompl_toSubmodule (I : LieIdeal K L) (hI : IsAtom I) :
    IsCompl I.toSubmodule (orthogonal Φ hΦ_inv I).toSubmodule := by
  rw [orthogonal_toSubmodule, LinearMap.BilinForm.isCompl_orthogonal_iff_disjoint hΦ_refl,
      ← orthogonal_toSubmodule _ hΦ_inv, ← LieSubmodule.disjoint_iff_toSubmodule]
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ Disjoint I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
  -/
  exact orthogonal_disjoint Φ hΦ_nondeg hΦ_inv hL I hI
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")]
alias orthogonal_isCompl_coe_submodule := orthogonal_isCompl_toSubmodule


open Module Submodule in
lemma orthogonal_isCompl (I : LieIdeal K L) (hI : IsAtom I) :
    IsCompl I (orthogonal Φ hΦ_inv I) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ IsCompl I (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
  -/
  rw [LieSubmodule.isCompl_iff_toSubmodule]
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ IsCompl ↑I ↑(LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
  -/
  exact orthogonal_isCompl_toSubmodule Φ hΦ_nondeg hΦ_inv hΦ_refl hL I hI
  /-
    🎉 no goals
  -/


lemma restrict_nondegenerate (I : LieIdeal K L) (hI : IsAtom I) :
    (Φ.restrict I).Nondegenerate := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ (Φ.restrict (LieIdeal.toLieSubalgebra K L I).toSubmodule).Nondegenerate
  -/
  rw [LinearMap.BilinForm.restrict_nondegenerate_iff_isCompl_orthogonal hΦ_refl]
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ IsCompl (LieIdeal.toLieSubalgebra K L I).toSubmodule (Φ.orthogonal (LieIdeal …
  -/
  exact orthogonal_isCompl_toSubmodule Φ hΦ_nondeg hΦ_inv hΦ_refl hL I hI
  /-
    🎉 no goals
  -/


lemma restrict_orthogonal_nondegenerate (I : LieIdeal K L) (hI : IsAtom I) :
    (Φ.restrict (orthogonal Φ hΦ_inv I)).Nondegenerate := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ (Φ.restrict (LieIdeal.toLieSubalgebra K L (LieAlgebra.InvariantForm.orthogon …
  -/
  rw [LinearMap.BilinForm.restrict_nondegenerate_iff_isCompl_orthogonal hΦ_refl]
  simp only [LieIdeal.toLieSubalgebra_toSubmodule, orthogonal_toSubmodule,
    LinearMap.BilinForm.orthogonal_orthogonal hΦ_nondeg hΦ_refl]
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : IsAtom I
    ⊢ IsCompl (Φ.orthogonal ↑I) ↑I
  -/
  exact (orthogonal_isCompl_toSubmodule Φ hΦ_nondeg hΦ_inv hΦ_refl hL I hI).symm
  /-
    🎉 no goals
  -/


open Module Submodule in
lemma atomistic : ∀ I : LieIdeal K L, sSup {J : LieIdeal K L | IsAtom J ∧ J ≤ I} = I := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    ⊢ ∀ (I : LieIdeal K L), Eq (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le  …
  -/
  intro I
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    ⊢ Eq (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le J I))) I
  -/
  apply le_antisymm
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra K L
      inst✝ : Module.Finite K L
      Φ : LinearMap.BilinForm K L
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      hΦ_refl : Φ.IsRefl
      hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
      I : LieIdeal K L
      ⊢ LE.le (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le J I))) I
    -/
  · apply sSup_le
    /-
      case a.a
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra K L
      inst✝ : Module.Finite K L
      Φ : LinearMap.BilinForm K L
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      hΦ_refl : Φ.IsRefl
      hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
      I : LieIdeal K L
      ⊢ ∀ (b : LieIdeal K L), Membership.mem (setOf fun J => And (IsAtom J) (LE.le J …
    -/
    rintro J ⟨-, hJ'⟩
    /-
      case a.a.intro
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra K L
      inst✝ : Module.Finite K L
      Φ : LinearMap.BilinForm K L
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      hΦ_refl : Φ.IsRefl
      hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
      I J : LieIdeal K L
      hJ' : LE.le J I
      ⊢ LE.le J I
    -/
    exact hJ'
    /-
      🎉 no goals
    -/
  /-
    case a
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    ⊢ LE.le I (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le J I)))
  -/
  by_cases hI : I = ⊥
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra K L
      inst✝ : Module.Finite K L
      Φ : LinearMap.BilinForm K L
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      hΦ_refl : Φ.IsRefl
      hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
      I : LieIdeal K L
      hI : Eq I Bot.bot
      ⊢ LE.le I (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le J I)))
    -/
  · exact hI.le.trans bot_le
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Not (Eq I Bot.bot)
    ⊢ LE.le I (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le J I)))
  -/
  obtain ⟨J, hJ, hJI⟩ := (eq_bot_or_exists_atom_le I).resolve_left hI
  /-
    case neg.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Not (Eq I Bot.bot)
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : LE.le J I
    ⊢ LE.le I (SupSet.sSup (setOf fun J => And (IsAtom J) (LE.le J I)))
  -/
  let J' := orthogonal Φ hΦ_inv J
  suffices I ≤ J ⊔ (J' ⊓ I) by
    refine this.trans ?_
    apply sup_le
    · exact le_sSup ⟨hJ, hJI⟩
    rw [← atomistic (J' ⊓ I)]
    apply sSup_le_sSup
    simp only [le_inf_iff, Set.setOf_subset_setOf, and_imp]
    tauto
  /-
    case neg.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Not (Eq I Bot.bot)
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : LE.le J I
    J' : LieSubmodule K L L := LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv J
    ⊢ LE.le I (Max.max J (Min.min J' I))
  -/
  suffices J ⊔ J' = ⊤ by rw [← sup_inf_assoc_of_le _ hJI, this, top_inf_eq]
  /-
    case neg.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Not (Eq I Bot.bot)
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : LE.le J I
    J' : LieSubmodule K L L := LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv J
    ⊢ Eq (Max.max J J') Top.top
  -/
  exact (orthogonal_isCompl Φ hΦ_nondeg hΦ_inv hΦ_refl hL J hJ).codisjoint.eq_top
  /-
    🎉 no goals
  -/
termination_by I => finrank K I
decreasing_by
  apply finrank_lt_finrank_of_lt
  suffices ¬I ≤ J' by simpa
  intro hIJ'
  apply hJ.1
  rw [eq_bot_iff]
  exact orthogonal_disjoint Φ hΦ_nondeg hΦ_inv hL J hJ le_rfl (hJI.trans hIJ')


open LieSubmodule in
/--
A finite-dimensional Lie algebra over a field is semisimple
if it does not have non-trivial abelian ideals and it admits a
non-degenerate reflexive invariant bilinear form.
Here a form is *invariant* if it is compatible with the Lie bracket: `Φ ⁅x, y⁆ z = Φ x ⁅y, z⁆`.
-/
theorem isSemisimple_of_nondegenerate : IsSemisimple K L := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    ⊢ LieAlgebra.IsSemisimple K L
  -/
  refine ⟨?_, ?_, hL⟩
    /-
      case refine_1
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra K L
      inst✝ : Module.Finite K L
      Φ : LinearMap.BilinForm K L
      hΦ_nondeg : Φ.Nondegenerate
      hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
      hΦ_refl : Φ.IsRefl
      hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
      ⊢ Eq (SupSet.sSup (setOf fun I => IsAtom I)) Top.top
    -/
  · simpa using atomistic Φ hΦ_nondeg hΦ_inv hΦ_refl hL ⊤
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    ⊢ sSupIndep (setOf fun I => IsAtom I)
  -/
  intro I hI
  /-
    case refine_2
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    ⊢ Disjoint I (SupSet.sSup (SDiff.sdiff (setOf fun I => IsAtom I) (Singleton.si …
  -/
  apply (orthogonal_disjoint Φ hΦ_nondeg hΦ_inv hL I hI).mono_right
  /-
    case refine_2
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    ⊢ LE.le (SupSet.sSup (SDiff.sdiff (setOf fun I => IsAtom I) (Singleton.singlet …
  -/
  apply sSup_le
  /-
    case refine_2.a
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    ⊢ ∀ (b : LieIdeal K L), Membership.mem (SDiff.sdiff (setOf fun I => IsAtom I)  …
  -/
  simp only [Set.mem_diff, Set.mem_setOf_eq, Set.mem_singleton_iff, and_imp]
  /-
    case refine_2.a
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    ⊢ ∀ (b : LieIdeal K L), IsAtom b → Not (Eq b I) → LE.le b (LieAlgebra.Invarian …
  -/
  intro J hJ hJI
  /-
    case refine_2.a
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    ⊢ LE.le J (LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)
  -/
  rw [← lie_eq_self_of_isAtom_of_nonabelian J hJ (hL J hJ), lieIdeal_oper_eq_span, lieSpan_le]
  /-
    case refine_2.a
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket …
  -/
  rintro _ ⟨x, y, rfl⟩
  /-
    case refine_2.a.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    ⊢ Membership.mem (↑(LieAlgebra.InvariantForm.orthogonal Φ hΦ_inv I)) (Bracket. …
  -/
  simp only [orthogonal_carrier, Φ.isOrtho_def, Set.mem_setOf_eq]
  /-
    case refine_2.a.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    ⊢ ∀ (n : L), Membership.mem I n → Eq ((Φ n) (Bracket.bracket ↑x ↑y)) 0
  -/
  intro z hz
  /-
    case refine_2.a.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    z : L
    hz : Membership.mem I z
    ⊢ Eq ((Φ z) (Bracket.bracket ↑x ↑y)) 0
  -/
  rw [← neg_eq_zero, ← hΦ_inv]
  /-
    case refine_2.a.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    z : L
    hz : Membership.mem I z
    ⊢ Eq ((Φ (Bracket.bracket (↑x) z)) ↑y) 0
  -/
  suffices ⁅(x : L), z⁆ = 0 by simp only [this, map_zero, LinearMap.zero_apply]
  /-
    case refine_2.a.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    z : L
    hz : Membership.mem I z
    ⊢ Eq (Bracket.bracket (↑x) z) 0
  -/
  rw [← LieSubmodule.mem_bot (R := K) (L := L), ← (hJ.disjoint_of_ne hI hJI).eq_bot]
  /-
    case refine_2.a.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    z : L
    hz : Membership.mem I z
    ⊢ Membership.mem (Min.min J I) (Bracket.bracket (↑x) z)
  -/
  apply lie_le_inf
  /-
    case refine_2.a.intro.intro.a
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    Φ : LinearMap.BilinForm K L
    hΦ_nondeg : Φ.Nondegenerate
    hΦ_inv : LinearMap.BilinForm.lieInvariant L Φ
    hΦ_refl : Φ.IsRefl
    hL : ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Memb …
    I : LieIdeal K L
    hI : Membership.mem (setOf fun I => IsAtom I) I
    J : LieIdeal K L
    hJ : IsAtom J
    hJI : Not (Eq J I)
    x y : Subtype fun x => Membership.mem J x
    z : L
    hz : Membership.mem I z
    ⊢ Membership.mem (Bracket.bracket J I) (Bracket.bracket (↑x) z)
  -/
  exact lie_mem_lie x.2 hz
  /-
    🎉 no goals
  -/


