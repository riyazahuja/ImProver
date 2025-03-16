theorem map_isUnit_of_le (hS : S ≤ A⁰) (s : S) : IsUnit (algebraMap A K s) := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    inst✝² : CommRing K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    hS : LE.le S (nonZeroDivisors A)
    s : Subtype fun x => Membership.mem S x
    ⊢ IsUnit ((algebraMap A K) ↑s)
  -/
  apply IsLocalization.map_units K (⟨s.1, hS s.2⟩ : A⁰)
  /-
    🎉 no goals
  -/


/-- The canonical map from a localization of `A` at `S` to the fraction ring
  of `A`, given that `S ≤ A⁰`. -/
noncomputable def mapToFractionRing (B : Type*) [CommRing B] [Algebra A B] [IsLocalization S B]
    (hS : S ≤ A⁰) : B →ₐ[A] K :=
                                                                                /-
                                                                                  A : Type u_1
                                                                                  K : Type u_2
                                                                                  inst✝⁶ : CommRing A
                                                                                  S : Submonoid A
                                                                                  hS✝ : LE.le S (nonZeroDivisors A)
                                                                                  inst✝⁵ : CommRing K
                                                                                  inst✝⁴ : Algebra A K
                                                                                  inst✝³ : IsFractionRing A K
                                                                                  B : Type u_3
                                                                                  inst✝² : CommRing B
                                                                                  inst✝¹ : Algebra A B
                                                                                  inst✝ : IsLocalization S B
                                                                                  hS : LE.le S (nonZeroDivisors A)
                                                                                  a : A
                                                                                  ⊢ Eq ((↑↑__src✝).toFun ((algebraMap A B) a)) ((algebraMap A K) a)
                                                                                -/
  { IsLocalization.lift (map_isUnit_of_le K S hS) with commutes' := fun a => by simp }
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
theorem mapToFractionRing_apply {B : Type*} [CommRing B] [Algebra A B] [IsLocalization S B]
    (hS : S ≤ A⁰) (b : B) :
    mapToFractionRing K S B hS b = IsLocalization.lift (map_isUnit_of_le K S hS) b :=
  rfl


theorem mem_range_mapToFractionRing_iff (B : Type*) [CommRing B] [Algebra A B] [IsLocalization S B]
    (hS : S ≤ A⁰) (x : K) :
    x ∈ (mapToFractionRing K S B hS).range ↔
      ∃ (a s : A) (hs : s ∈ S), x = IsLocalization.mk' K a ⟨s, hS hs⟩ :=
  ⟨by
    /-
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      x : K
      ⊢ Membership.mem (Localization.mapToFractionRing K S B hS).range x → Exists fu …
    -/
    rintro ⟨x, rfl⟩
    /-
      case intro
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      x : B
      ⊢ Exists fun a => Exists fun s => Exists fun hs => Eq ((Localization.mapToFrac …
    -/
    obtain ⟨a, s, rfl⟩ := IsLocalization.mk'_surjective S x
    /-
      case intro.intro.intro
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      a : A
      s : Subtype fun x => Membership.mem S x
      ⊢ Exists fun a_1 => Exists fun s_1 => Exists fun hs => Eq ((Localization.mapTo …
    -/
    use a, s, s.2
    /-
      case h
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      a : A
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq ((Localization.mapToFractionRing K S B hS).toRingHom (IsLocalization.mk'  …
    -/
    apply IsLocalization.lift_mk', by
    /-
      🎉 no goals
    -/
    /-
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      x : K
      ⊢ (Exists fun a => Exists fun s => Exists fun hs => Eq x (IsLocalization.mk' K …
    -/
    rintro ⟨a, s, hs, rfl⟩
    /-
      case intro.intro.intro
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      a s : A
      hs : Membership.mem S s
      ⊢ Membership.mem (Localization.mapToFractionRing K S B hS).range (IsLocalizati …
    -/
    use IsLocalization.mk' _ a ⟨s, hs⟩
    /-
      case h
      A : Type u_1
      K : Type u_2
      inst✝⁶ : CommRing A
      S : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsFractionRing A K
      B : Type u_3
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsLocalization S B
      hS : LE.le S (nonZeroDivisors A)
      a s : A
      hs : Membership.mem S s
      ⊢ Eq ((Localization.mapToFractionRing K S B hS).toRingHom (IsLocalization.mk'  …
    -/
    apply IsLocalization.lift_mk'⟩
    /-
      🎉 no goals
    -/


instance isLocalization_range_mapToFractionRing (B : Type*) [CommRing B] [Algebra A B]
    [IsLocalization S B] (hS : S ≤ A⁰) : IsLocalization S (mapToFractionRing K S B hS).range :=
  IsLocalization.isLocalization_of_algEquiv S <|
    show B ≃ₐ[A] _ from AlgEquiv.ofBijective (mapToFractionRing K S B hS).rangeRestrict (by
      /-
        A : Type u_1
        K : Type u_2
        inst✝⁶ : CommRing A
        S : Submonoid A
        hS✝ : LE.le S (nonZeroDivisors A)
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsFractionRing A K
        B : Type u_3
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : IsLocalization S B
        hS : LE.le S (nonZeroDivisors A)
        ⊢ Function.Bijective ⇑(Localization.mapToFractionRing K S B hS).rangeRestrict
      -/
      refine ⟨fun a b h => ?_, Set.surjective_onto_range⟩
      /-
        A : Type u_1
        K : Type u_2
        inst✝⁶ : CommRing A
        S : Submonoid A
        hS✝ : LE.le S (nonZeroDivisors A)
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsFractionRing A K
        B : Type u_3
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : IsLocalization S B
        hS : LE.le S (nonZeroDivisors A)
        a b : B
        h : Eq ((Localization.mapToFractionRing K S B hS).rangeRestrict a) ((Localizat …
        ⊢ Eq a b
      -/
      refine (IsLocalization.lift_injective_iff _).2 (fun a b => ?_) (Subtype.ext_iff.1 h)
      exact ⟨fun h => congr_arg _ (IsLocalization.injective _ hS h),
        fun h => congr_arg _ (IsFractionRing.injective A K h)⟩)


instance isFractionRing_range_mapToFractionRing (B : Type*) [CommRing B] [Algebra A B]
    [IsLocalization S B] (hS : S ≤ A⁰) : IsFractionRing (mapToFractionRing K S B hS).range K :=
  IsFractionRing.isFractionRing_of_isLocalization S _ _ hS


/-- Given a commutative ring `A` with fraction ring `K`, and a submonoid `S` of `A` which
contains no zero divisor, this is the localization of `A` at `S`, considered as
a subalgebra of `K` over `A`.

The carrier of this subalgebra is defined as the set of all `x : K` of the form
`IsLocalization.mk' K a ⟨s, _⟩`, where `s ∈ S`.
-/
noncomputable def subalgebra (hS : S ≤ A⁰) : Subalgebra A K :=
  (mapToFractionRing K S (Localization S) hS).range.copy
      { x | ∃ (a s : A) (hs : s ∈ S), x = IsLocalization.mk' K a ⟨s, hS hs⟩ } <| by
    /-
      A : Type u_1
      K : Type u_2
      inst✝³ : CommRing A
      S : Submonoid A
      hS✝ : LE.le S (nonZeroDivisors A)
      inst✝² : CommRing K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      hS : LE.le S (nonZeroDivisors A)
      ⊢ Eq (setOf fun x => Exists fun a => Exists fun s => Exists fun hs => Eq x (Is …
    -/
    ext
    /-
      case h
      A : Type u_1
      K : Type u_2
      inst✝³ : CommRing A
      S : Submonoid A
      hS✝ : LE.le S (nonZeroDivisors A)
      inst✝² : CommRing K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      hS : LE.le S (nonZeroDivisors A)
      x✝ : K
      ⊢ Iff (Membership.mem (setOf fun x => Exists fun a => Exists fun s => Exists f …
    -/
    symm
    /-
      case h
      A : Type u_1
      K : Type u_2
      inst✝³ : CommRing A
      S : Submonoid A
      hS✝ : LE.le S (nonZeroDivisors A)
      inst✝² : CommRing K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      hS : LE.le S (nonZeroDivisors A)
      x✝ : K
      ⊢ Iff (Membership.mem (↑(Localization.mapToFractionRing K S (Localization S) h …
    -/
    apply mem_range_mapToFractionRing_iff
    /-
      🎉 no goals
    -/


instance isLocalization_subalgebra : IsLocalization S (subalgebra K S hS) := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝² : CommRing K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsLocalization S (Subtype fun x => Membership.mem (Localization.subalgebra K …
  -/
  dsimp only [Localization.subalgebra]
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝² : CommRing K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsLocalization S (Subtype fun x => Membership.mem ((Localization.mapToFracti …
  -/
  rw [Subalgebra.copy_eq]
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝² : CommRing K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsLocalization S (Subtype fun x => Membership.mem (Localization.mapToFractio …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isFractionRing : IsFractionRing (subalgebra K S hS) K :=
  IsFractionRing.isFractionRing_of_isLocalization S _ _ hS


theorem mem_range_mapToFractionRing_iff_ofField (B : Type*) [CommRing B] [Algebra A B]
    [IsLocalization S B] (x : K) :
    x ∈ (mapToFractionRing K S B hS).range ↔
      ∃ (a s : A) (_ : s ∈ S), x = algebraMap A K a * (algebraMap A K s)⁻¹ := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁶ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝⁵ : Field K
    inst✝⁴ : Algebra A K
    inst✝³ : IsFractionRing A K
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsLocalization S B
    x : K
    ⊢ Iff (Membership.mem (Localization.mapToFractionRing K S B hS).range x) (Exis …
  -/
  rw [mem_range_mapToFractionRing_iff]
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁶ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝⁵ : Field K
    inst✝⁴ : Algebra A K
    inst✝³ : IsFractionRing A K
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsLocalization S B
    x : K
    ⊢ Iff (Exists fun a => Exists fun s => Exists fun hs => Eq x (IsLocalization.m …
  -/
  convert Iff.rfl
  /-
    case h.e'_2.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_3
    A : Type u_1
    K : Type u_2
    inst✝⁶ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝⁵ : Field K
    inst✝⁴ : Algebra A K
    inst✝³ : IsFractionRing A K
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsLocalization S B
    x : K
    x✝² x✝¹ : A
    x✝ : Membership.mem S x✝¹
    ⊢ Eq (HMul.hMul ((algebraMap A K) x✝²) (Inv.inv ((algebraMap A K) x✝¹))) (IsLo …
  -/
  congr
  /-
    case h.e'_2.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_3.e_a
    A : Type u_1
    K : Type u_2
    inst✝⁶ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝⁵ : Field K
    inst✝⁴ : Algebra A K
    inst✝³ : IsFractionRing A K
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsLocalization S B
    x : K
    x✝² x✝¹ : A
    x✝ : Membership.mem S x✝¹
    ⊢ Eq (Inv.inv ((algebraMap A K) x✝¹)) ↑(Inv.inv ((IsUnit.liftRight ((IsLocaliz …
  -/
  rw [Units.val_inv_eq_inv_val]
  /-
    case h.e'_2.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_3.e_a
    A : Type u_1
    K : Type u_2
    inst✝⁶ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝⁵ : Field K
    inst✝⁴ : Algebra A K
    inst✝³ : IsFractionRing A K
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsLocalization S B
    x : K
    x✝² x✝¹ : A
    x✝ : Membership.mem S x✝¹
    ⊢ Eq (Inv.inv ((algebraMap A K) x✝¹)) (Inv.inv ↑((IsUnit.liftRight ((IsLocaliz …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a domain `A` with fraction field `K`, and a submonoid `S` of `A` which
contains no zero divisor, this is the localization of `A` at `S`, considered as
a subalgebra of `K` over `A`.

The carrier of this subalgebra is defined as the set of all `x : K` of the form
`algebraMap A K a * (algebraMap A K s)⁻¹` where `a s : A` and `s ∈ S`.
-/
noncomputable def ofField : Subalgebra A K :=
  (mapToFractionRing K S (Localization S) hS).range.copy
      { x | ∃ (a s : A) (_ : s ∈ S), x = algebraMap A K a * (algebraMap A K s)⁻¹ } <| by
    /-
      A : Type u_1
      K : Type u_2
      inst✝³ : CommRing A
      S : Submonoid A
      hS : LE.le S (nonZeroDivisors A)
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      ⊢ Eq (setOf fun x => Exists fun a => Exists fun s => Exists fun x_1 => Eq x (H …
    -/
    ext
    /-
      case h
      A : Type u_1
      K : Type u_2
      inst✝³ : CommRing A
      S : Submonoid A
      hS : LE.le S (nonZeroDivisors A)
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x✝ : K
      ⊢ Iff (Membership.mem (setOf fun x => Exists fun a => Exists fun s => Exists f …
    -/
    symm
    /-
      case h
      A : Type u_1
      K : Type u_2
      inst✝³ : CommRing A
      S : Submonoid A
      hS : LE.le S (nonZeroDivisors A)
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x✝ : K
      ⊢ Iff (Membership.mem (↑(Localization.mapToFractionRing K S (Localization S) h …
    -/
    apply mem_range_mapToFractionRing_iff_ofField
    /-
      🎉 no goals
    -/


instance isLocalization_ofField : IsLocalization S (subalgebra.ofField K S hS) := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsLocalization S (Subtype fun x => Membership.mem (Localization.subalgebra.o …
  -/
  dsimp only [Localization.subalgebra.ofField]
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsLocalization S (Subtype fun x => Membership.mem ((Localization.mapToFracti …
  -/
  rw [Subalgebra.copy_eq]
  /-
    A : Type u_1
    K : Type u_2
    inst✝³ : CommRing A
    S : Submonoid A
    hS : LE.le S (nonZeroDivisors A)
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsLocalization S (Subtype fun x => Membership.mem (Localization.mapToFractio …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isFractionRing_ofField : IsFractionRing (subalgebra.ofField K S hS) K :=
  IsFractionRing.isFractionRing_of_isLocalization S _ _ hS


