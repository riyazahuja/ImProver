instance [Nontrivial S] (f : R →+* S) (s : Subring R) [IsLocalRing s] : IsLocalRing (s.map f) :=
  .of_surjective' (f.restrict s _ (fun _ ↦ Set.mem_image_of_mem f))
    (fun ⟨_, a, ha, e⟩ ↦ ⟨⟨a, ha⟩, Subtype.ext e⟩)


instance isLocalRing_top [IsLocalRing R] : IsLocalRing (⊤ : Subring R) :=
  Subring.topEquiv.symm.isLocalRing


variable (R) in
/-- The class of local subrings of a commutative ring. -/
@[ext]
structure LocalSubring where
  /-- The underlying subring of a local subring. -/
  toSubring : Subring R
  [isLocalRing : IsLocalRing toSubring]


lemma toSubring_injective : Function.Injective (toSubring (R := R)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Function.Injective LocalSubring.toSubring
  -/
  rintro ⟨a, b⟩ ⟨c, d⟩ rfl; rfl
                            /-
                              🎉 no goals
                            -/


/-- Copy of a local subring with a new `carrier` equal to the old one.
Useful to fix definitional equalities. -/
protected def copy (S : LocalSubring R) (s : Set R) (hs : s = ↑S.toSubring) : LocalSubring R :=
  LocalSubring.mk (S.toSubring.copy s hs) (isLocalRing := hs ▸ S.2)


/-- The image of a `LocalSubring` as a `LocalSubring`. -/
@[simps! toSubring]
def map [Nontrivial S] (f : R →+* S) (s : LocalSubring R) : LocalSubring S :=
  mk (s.1.map f)


/-- The range of a ringhom from a local ring as a `LocalSubring`. -/
@[simps! toSubring]
def range [IsLocalRing R] [Nontrivial S] (f : R →+* S) : LocalSubring S :=
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     inst✝⁴ : CommRing R
                                     inst✝³ : CommRing S
                                     K : Type u_3
                                     inst✝² : Field K
                                     inst✝¹ : IsLocalRing R
                                     inst✝ : Nontrivial S
                                     f : RingHom R S
                                     ⊢ Eq ↑f.range ↑(LocalSubring.map f (LocalSubring.mk Top.top)).toSubring
                                   -/
  .copy (map f (mk ⊤)) f.range (by ext x; exact congr(x ∈ $(Set.image_univ.symm)))
                                          /-
                                            🎉 no goals
                                          -/


/--
The domination order on local subrings.
`A` dominates `B` if and only if `B ≤ A` (as subrings) and `m_A ∩ B = m_B`.
-/
@[stacks 00I9]
instance : PartialOrder (LocalSubring R) where
  le A B := ∃ h : A.1 ≤ B.1, IsLocalHom (Subring.inclusion h)
  le_refl a := ⟨le_rfl, ⟨fun _ ↦ id⟩⟩
  le_trans A B C h₁ h₂ := ⟨h₁.1.trans h₂.1, @RingHom.isLocalHom_comp _ _ _ _ _ _ _ _ h₂.2 h₁.2⟩
  le_antisymm A B h₁ h₂ := toSubring_injective (le_antisymm h₁.1 h₂.1)


/-- `A` dominates `B` if and only if `B ≤ A` (as subrings) and `m_A ∩ B = m_B`. -/
lemma le_def {A B : LocalSubring R} :
    A ≤ B ↔ ∃ h : A.toSubring ≤ B.toSubring, IsLocalHom (Subring.inclusion h) := Iff.rfl


lemma toSubring_mono : Monotone (toSubring (R := R)) :=
  fun _ _ e ↦ e.1


/-- The localization of a subring at a prime, as a local subring.
Also see `Localization.subalgebra.ofField` -/
noncomputable
def ofPrime (A : Subring K) (P : Ideal A) [P.IsPrime] : LocalSubring K :=
  range (IsLocalization.lift (M := P.primeCompl) (S := Localization.AtPrime P)
                         /-
                           R : Type u_1
                           S : Type u_2
                           inst✝⁴ : CommRing R
                           inst✝³ : CommRing S
                           K : Type u_3
                           inst✝² : Field K
                           A✝ : Subring K
                           P✝ : Ideal (Subtype fun x => Membership.mem A✝ x)
                           inst✝¹ : P✝.IsPrime
                           A : Subring K
                           P : Ideal (Subtype fun x => Membership.mem A x)
                           inst✝ : P.IsPrime
                           ⊢ ∀ (y : Subtype fun x => Membership.mem P.primeCompl x), IsUnit (A.subtype ↑y)
                         -/
    (g := A.subtype) (by simp [Ideal.primeCompl, not_imp_not]))
                         /-
                           🎉 no goals
                         -/


lemma le_ofPrime : A ≤ (ofPrime A P).toSubring := by
  /-
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    ⊢ LE.le A (LocalSubring.ofPrime A P).toSubring
  -/
  intro x hx
  /-
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x : K
    hx : Membership.mem A x
    ⊢ Membership.mem (LocalSubring.ofPrime A P).toSubring x
  -/
  exact ⟨algebraMap A _ ⟨x, hx⟩, by simp⟩
  /-
    🎉 no goals
  -/


noncomputable
instance : Algebra A (ofPrime A P).toSubring := (Subring.inclusion (le_ofPrime A P)).toAlgebra


instance : IsScalarTower A (ofPrime A P).toSubring K := .of_algebraMap_eq (fun _ ↦ rfl)


/-- The localization of a subring at a prime is indeed isomorphic to its abstract localization. -/
noncomputable
def ofPrimeEquiv : Localization.AtPrime P ≃ₐ[A] (ofPrime A P).toSubring := by
  refine AlgEquiv.ofInjective (IsLocalization.liftAlgHom (M := P.primeCompl)
    (S := Localization.AtPrime P) (f := Algebra.ofId A K) _) ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    ⊢ Function.Injective ⇑(IsLocalization.liftAlgHom ⋯)
  -/
  intro x y e
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x y : Localization.AtPrime P
    e : Eq ((IsLocalization.liftAlgHom ⋯) x) ((IsLocalization.liftAlgHom ⋯) y)
    ⊢ Eq x y
  -/
  obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective P.primeCompl x
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    y : Localization.AtPrime P
    x : Subtype fun x => Membership.mem A x
    s : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq ((IsLocalization.liftAlgHom ⋯) (IsLocalization.mk' (Localization.AtPrim …
    ⊢ Eq (IsLocalization.mk' (Localization.AtPrime P) x s) y
  -/
  obtain ⟨y, t, rfl⟩ := IsLocalization.mk'_surjective P.primeCompl y
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x : Subtype fun x => Membership.mem A x
    s : Subtype fun x => Membership.mem P.primeCompl x
    y : Subtype fun x => Membership.mem A x
    t : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq ((IsLocalization.liftAlgHom ⋯) (IsLocalization.mk' (Localization.AtPrim …
    ⊢ Eq (IsLocalization.mk' (Localization.AtPrime P) x s) (IsLocalization.mk' (Lo …
  -/
  have H (x : P.primeCompl) : x.1 ≠ 0 := by aesop
  have : x.1 = y.1 * t.1.1⁻¹ * s.1.1 := by
    simpa [IsLocalization.lift_mk', Algebra.ofId_apply, H,
      Algebra.algebraMap_ofSubring_apply, IsUnit.coe_liftRight] using congr($e * s.1.1)
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x : Subtype fun x => Membership.mem A x
    s : Subtype fun x => Membership.mem P.primeCompl x
    y : Subtype fun x => Membership.mem A x
    t : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq ((IsLocalization.liftAlgHom ⋯) (IsLocalization.mk' (Localization.AtPrim …
    H : ∀ (x : Subtype fun x => Membership.mem P.primeCompl x), Ne (↑x) 0
    this : Eq (↑x) (HMul.hMul (HMul.hMul (↑y) (Inv.inv ↑↑t)) ↑↑s)
    ⊢ Eq (IsLocalization.mk' (Localization.AtPrime P) x s) (IsLocalization.mk' (Lo …
  -/
  rw [IsLocalization.mk'_eq_iff_eq]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x : Subtype fun x => Membership.mem A x
    s : Subtype fun x => Membership.mem P.primeCompl x
    y : Subtype fun x => Membership.mem A x
    t : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq ((IsLocalization.liftAlgHom ⋯) (IsLocalization.mk' (Localization.AtPrim …
    H : ∀ (x : Subtype fun x => Membership.mem P.primeCompl x), Ne (↑x) 0
    this : Eq (↑x) (HMul.hMul (HMul.hMul (↑y) (Inv.inv ↑↑t)) ↑↑s)
    ⊢ Eq ((algebraMap (Subtype fun x => Membership.mem A x) (Localization.AtPrime  …
  -/
  congr 1
  /-
    case intro.intro.intro.intro.h.e_6.h
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x : Subtype fun x => Membership.mem A x
    s : Subtype fun x => Membership.mem P.primeCompl x
    y : Subtype fun x => Membership.mem A x
    t : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq ((IsLocalization.liftAlgHom ⋯) (IsLocalization.mk' (Localization.AtPrim …
    H : ∀ (x : Subtype fun x => Membership.mem P.primeCompl x), Ne (↑x) 0
    this : Eq (↑x) (HMul.hMul (HMul.hMul (↑y) (Inv.inv ↑↑t)) ↑↑s)
    ⊢ Eq (HMul.hMul (↑t) x) (HMul.hMul (↑s) y)
  -/
  ext
  /-
    case intro.intro.intro.intro.h.e_6.h.a
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    K : Type u_3
    inst✝¹ : Field K
    A : Subring K
    P : Ideal (Subtype fun x => Membership.mem A x)
    inst✝ : P.IsPrime
    x : Subtype fun x => Membership.mem A x
    s : Subtype fun x => Membership.mem P.primeCompl x
    y : Subtype fun x => Membership.mem A x
    t : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq ((IsLocalization.liftAlgHom ⋯) (IsLocalization.mk' (Localization.AtPrim …
    H : ∀ (x : Subtype fun x => Membership.mem P.primeCompl x), Ne (↑x) 0
    this : Eq (↑x) (HMul.hMul (HMul.hMul (↑y) (Inv.inv ↑↑t)) ↑↑s)
    ⊢ Eq ↑(HMul.hMul (↑t) x) ↑(HMul.hMul (↑s) y)
  -/
  field_simp [H t, this, mul_comm]
  /-
    🎉 no goals
  -/


instance : IsLocalization.AtPrime (ofPrime A P).toSubring P :=
  IsLocalization.isLocalization_of_algEquiv _ (ofPrimeEquiv A P)


