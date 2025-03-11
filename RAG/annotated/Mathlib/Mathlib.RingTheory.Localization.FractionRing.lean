/-- `IsFractionRing R K` states `K` is the ring of fractions of a commutative ring `R`. -/
abbrev IsFractionRing [CommRing K] [Algebra R K] :=
  IsLocalization (nonZeroDivisors R) K


instance {R : Type*} [Field R] : IsFractionRing R R :=
  IsLocalization.at_units _ (fun _ ↦ isUnit_of_mem_nonZeroDivisors)


/-- The cast from `Int` to `Rat` as a `FractionRing`. -/
instance Rat.isFractionRing : IsFractionRing ℤ ℚ where
  map_units' := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      ⊢ ∀ (y : Subtype fun x => Membership.mem (nonZeroDivisors Int) x), IsUnit ((al …
    -/
    rintro ⟨x, hx⟩
    /-
      case mk
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      x : Int
      hx : Membership.mem (nonZeroDivisors Int) x
      ⊢ IsUnit ((algebraMap Int Rat) ↑⟨x, hx⟩)
    -/
    rw [mem_nonZeroDivisors_iff_ne_zero] at hx
    /-
      case mk
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      x : Int
      hx✝ : Membership.mem (nonZeroDivisors Int) x
      hx : Ne x 0
      ⊢ IsUnit ((algebraMap Int Rat) ↑⟨x, hx✝⟩)
    -/
    simpa only [eq_intCast, isUnit_iff_ne_zero, Int.cast_eq_zero, Ne, Subtype.coe_mk] using hx
    /-
      🎉 no goals
    -/
  surj' := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      ⊢ ∀ (z : Rat), Exists fun x => Eq (HMul.hMul z ((algebraMap Int Rat) ↑x.2)) (( …
    -/
    rintro ⟨n, d, hd, h⟩
    /-
      case mk'
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      n : Int
      d : Nat
      hd : Ne d 0
      h : n.natAbs.Coprime d
      ⊢ Exists fun x => Eq (HMul.hMul { num := n, den := d, den_nz := hd, reduced := …
    -/
    refine ⟨⟨n, ⟨d, ?_⟩⟩, Rat.mul_den_eq_num _⟩
    /-
      case mk'
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      n : Int
      d : Nat
      hd : Ne d 0
      h : n.natAbs.Coprime d
      ⊢ Membership.mem (nonZeroDivisors Int) ↑d
    -/
    rw [mem_nonZeroDivisors_iff_ne_zero, Int.natCast_ne_zero_iff_pos]
    /-
      case mk'
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      n : Int
      d : Nat
      hd : Ne d 0
      h : n.natAbs.Coprime d
      ⊢ LT.lt 0 d
    -/
    exact Nat.zero_lt_of_ne_zero hd
    /-
      🎉 no goals
    -/
  exists_of_eq {x y} := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      x y : Int
      ⊢ Eq ((algebraMap Int Rat) x) ((algebraMap Int Rat) y) → Exists fun c => Eq (H …
    -/
    rw [eq_intCast, eq_intCast, Int.cast_inj]
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      x y : Int
      ⊢ Eq x y → Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    rintro rfl
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommRing P
      A : Type u_4
      inst✝ : CommRing A
      K : Type u_5
      x : Int
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) x)
    -/
    use 1
    /-
      🎉 no goals
    -/


theorem to_map_eq_zero_iff {x : R} : algebraMap R K x = 0 ↔ x = 0 :=
  IsLocalization.to_map_eq_zero_iff _ le_rfl


protected theorem injective : Function.Injective (algebraMap R K) :=
  IsLocalization.injective _ (le_of_eq rfl)


@[norm_cast, simp]
-- Porting note: using `↑` didn't work, so I needed to explicitly put in the cast myself
theorem coe_inj {a b : R} : (Algebra.cast a : K) = Algebra.cast b ↔ a = b :=
  (IsFractionRing.injective R K).eq_iff


instance (priority := 100) [NoZeroDivisors K] : NoZeroSMulDivisors R K :=
  NoZeroSMulDivisors.of_algebraMap_injective <| IsFractionRing.injective R K


protected theorem to_map_ne_zero_of_mem_nonZeroDivisors [Nontrivial R] {x : R}
    (hx : x ∈ nonZeroDivisors R) : algebraMap R K x ≠ 0 :=
  IsLocalization.to_map_ne_zero_of_mem_nonZeroDivisors _ le_rfl hx


include A in
/-- A `CommRing` `K` which is the localization of an integral domain `R` at `R - {0}` is an
integral domain. -/
protected theorem isDomain : IsDomain K :=
  isDomain_of_le_nonZeroDivisors _ (le_refl (nonZeroDivisors A))


/-- The inverse of an element in the field of fractions of an integral domain. -/
protected noncomputable irreducible_def inv (z : K) : K := open scoped Classical in
  if h : z = 0 then 0
  else
    mk' K ↑(sec (nonZeroDivisors A) z).2
      ⟨(sec _ z).1,
        mem_nonZeroDivisors_iff_ne_zero.2 fun h0 =>
          h <| eq_zero_of_fst_eq_zero (sec_spec (nonZeroDivisors A) z) h0⟩


protected theorem mul_inv_cancel (x : K) (hx : x ≠ 0) : x * IsFractionRing.inv A x = 1 := by
  rw [IsFractionRing.inv, dif_neg hx, ←
    IsUnit.mul_left_inj
      (map_units K
        ⟨(sec _ x).1,
          mem_nonZeroDivisors_iff_ne_zero.2 fun h0 =>
            hx <| eq_zero_of_fst_eq_zero (sec_spec (nonZeroDivisors A) x) h0⟩),
    one_mul, mul_assoc]
  /-
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : CommRing K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDomain A
    x : K
    hx : Ne x 0
    ⊢ Eq (HMul.hMul x (HMul.hMul (IsLocalization.mk' K ↑(IsLocalization.sec (nonZe …
  -/
  rw [mk'_spec, ← eq_mk'_iff_mul_eq]
  /-
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : CommRing K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDomain A
    x : K
    hx : Ne x 0
    ⊢ Eq x (IsLocalization.mk' K (↑⟨(IsLocalization.sec (nonZeroDivisors A) x).1,  …
  -/
  exact (mk'_sec _ x).symm
  /-
    🎉 no goals
  -/


/-- A `CommRing` `K` which is the localization of an integral domain `R` at `R - {0}` is a field.
See note [reducible non-instances]. -/
@[stacks 09FJ]
noncomputable abbrev toField : Field K where
  __ := IsFractionRing.isDomain A
  mul_inv_cancel := IsFractionRing.mul_inv_cancel A
                                                       /-
                                                         R : Type u_1
                                                         inst✝¹⁰ : CommRing R
                                                         M : Submonoid R
                                                         S : Type u_2
                                                         inst✝⁹ : CommRing S
                                                         inst✝⁸ : Algebra R S
                                                         P : Type u_3
                                                         inst✝⁷ : CommRing P
                                                         A : Type u_4
                                                         inst✝⁶ : CommRing A
                                                         K : Type u_5
                                                         inst✝⁵ : CommRing K
                                                         inst✝⁴ : Algebra R K
                                                         inst✝³ : IsFractionRing R K
                                                         inst✝² : Algebra A K
                                                         inst✝¹ : IsFractionRing A K
                                                         inst✝ : IsDomain A
                                                         ⊢ Eq (IsFractionRing.inv A 0) 0
                                                       -/
  inv_zero := show IsFractionRing.inv A (0 : K) = 0 by rw [IsFractionRing.inv]; exact dif_pos rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


lemma surjective_iff_isField [IsDomain R] : Function.Surjective (algebraMap R K) ↔ IsField R where
  mp h := (RingEquiv.ofBijective (algebraMap R K)
      ⟨IsFractionRing.injective R K, h⟩).toMulEquiv.isField _ (IsFractionRing.toField R).toIsField
  mpr h :=
    letI := h.toField
    (IsLocalization.atUnits R _ (S := K)
      (fun _ hx ↦ Ne.isUnit (mem_nonZeroDivisors_iff_ne_zero.mp hx))).surjective


theorem mk'_mk_eq_div {r s} (hs : s ∈ nonZeroDivisors A) :
    mk' K r ⟨s, hs⟩ = algebraMap A K r / algebraMap A K s :=
  haveI := (algebraMap A K).domain_nontrivial
  mk'_eq_iff_eq_mul.2 <|
    (div_mul_cancel₀ (algebraMap A K r)
        (IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors hs)).symm


@[simp]
theorem mk'_eq_div {r} (s : nonZeroDivisors A) : mk' K r s = algebraMap A K r / algebraMap A K s :=
  mk'_mk_eq_div s.2


theorem div_surjective (z : K) :
    ∃ x y : A, y ∈ nonZeroDivisors A ∧ algebraMap _ _ x / algebraMap _ _ y = z :=
  let ⟨x, ⟨y, hy⟩, h⟩ := mk'_surjective (nonZeroDivisors A) z
                /-
                  A : Type u_4
                  inst✝³ : CommRing A
                  K : Type u_5
                  inst✝² : Field K
                  inst✝¹ : Algebra A K
                  inst✝ : IsFractionRing A K
                  z : K
                  x y : A
                  hy : Membership.mem (nonZeroDivisors A) y
                  h : Eq (IsLocalization.mk' K x ⟨y, hy⟩) z
                  ⊢ Eq (HDiv.hDiv ((algebraMap A K) x) ((algebraMap A K) y)) z
                -/
  ⟨x, y, hy, by rwa [mk'_eq_div] at h⟩
                /-
                  🎉 no goals
                -/


theorem isUnit_map_of_injective (hg : Function.Injective g) (y : nonZeroDivisors A) :
    IsUnit (g y) :=
  haveI := g.domain_nontrivial
  IsUnit.mk0 (g y) <|
    show g.toMonoidWithZeroHom y ≠ 0 from map_ne_zero_of_mem_nonZeroDivisors g hg y.2


theorem mk'_eq_zero_iff_eq_zero [Algebra R K] [IsFractionRing R K] {x : R} {y : nonZeroDivisors R} :
    mk' K x y = 0 ↔ x = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : R
    y : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ Iff (Eq (IsLocalization.mk' K x y) 0) (Eq x 0)
  -/
  haveI := (algebraMap R K).domain_nontrivial
  /-
    R : Type u_1
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : R
    y : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    this : Nontrivial R
    ⊢ Iff (Eq (IsLocalization.mk' K x y) 0) (Eq x 0)
  -/
  simp [nonZeroDivisors.ne_zero]
  /-
    🎉 no goals
  -/


theorem mk'_eq_one_iff_eq {x : A} {y : nonZeroDivisors A} : mk' K x y = 1 ↔ x = y := by
  /-
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : A
    y : Subtype fun x => Membership.mem (nonZeroDivisors A) x
    ⊢ Iff (Eq (IsLocalization.mk' K x y) 1) (Eq x ↑y)
  -/
  haveI := (algebraMap A K).domain_nontrivial
  /-
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : A
    y : Subtype fun x => Membership.mem (nonZeroDivisors A) x
    this : Nontrivial A
    ⊢ Iff (Eq (IsLocalization.mk' K x y) 1) (Eq x ↑y)
  -/
  refine ⟨?_, fun hxy => by rw [hxy, mk'_self']⟩
  /-
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : A
    y : Subtype fun x => Membership.mem (nonZeroDivisors A) x
    this : Nontrivial A
    ⊢ Eq (IsLocalization.mk' K x y) 1 → Eq x ↑y
  -/
  intro hxy
  have hy : (algebraMap A K) ↑y ≠ (0 : K) :=
    IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors y.property
  /-
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : A
    y : Subtype fun x => Membership.mem (nonZeroDivisors A) x
    this : Nontrivial A
    hxy : Eq (IsLocalization.mk' K x y) 1
    hy : Ne ((algebraMap A K) ↑y) 0
    ⊢ Eq x ↑y
  -/
  rw [IsFractionRing.mk'_eq_div, div_eq_one_iff_eq hy] at hxy
  /-
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : A
    y : Subtype fun x => Membership.mem (nonZeroDivisors A) x
    this : Nontrivial A
    hxy : Eq ((algebraMap A K) x) ((algebraMap A K) ↑y)
    hy : Ne ((algebraMap A K) ↑y) 0
    ⊢ Eq x ↑y
  -/
  exact IsFractionRing.injective A K hxy
  /-
    🎉 no goals
  -/


variable (A K) in
/-- If `A` is a commutative ring with fraction field `K`, then the subfield of `K` generated by
the image of `algebraMap A K` is equal to the whole field `K`. -/
theorem closure_range_algebraMap : Subfield.closure (Set.range (algebraMap A K)) = ⊤ :=
  top_unique fun z _ ↦ by
    /-
      A : Type u_4
      inst✝³ : CommRing A
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      z : K
      x✝ : Membership.mem Top.top z
      ⊢ Membership.mem (Subfield.closure (Set.range ⇑(algebraMap A K))) z
    -/
    obtain ⟨_, _, -, rfl⟩ := div_surjective (A := A) z
    /-
      case intro.intro.intro
      A : Type u_4
      inst✝³ : CommRing A
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      w✝¹ w✝ : A
      x✝ : Membership.mem Top.top (HDiv.hDiv ((algebraMap A K) w✝¹) ((algebraMap A K …
      ⊢ Membership.mem (Subfield.closure (Set.range ⇑(algebraMap A K))) (HDiv.hDiv ( …
    -/
                      /-
                        🎉 no goals
                      -/
    apply div_mem <;> exact Subfield.subset_closure ⟨_, rfl⟩
                      /-
                        🎉 no goals
                      -/


/-- If `A` is a commutative ring with fraction field `K`, `L` is a field, `g : A →+* L` lifts to
`f : K →+* L`, then the image of `f` is the subfield generated by the image of `g`. -/
theorem ringHom_fieldRange_eq_of_comp_eq (h : RingHom.comp f (algebraMap A K) = g) :
    f.fieldRange = Subfield.closure g.range := by
  rw [f.fieldRange_eq_map, ← closure_range_algebraMap A K,
    f.map_field_closure, ← Set.range_comp, ← f.coe_comp, h, g.coe_range]


/-- If `A` is a commutative ring with fraction field `K`, `L` is a field, `g : A →+* L` lifts to
`f : K →+* L`, `s` is a set such that the image of `g` is the subring generated by `s`,
then the image of `f` is the subfield generated by `s`. -/
theorem ringHom_fieldRange_eq_of_comp_eq_of_range_eq (h : RingHom.comp f (algebraMap A K) = g)
    {s : Set L} (hs : g.range = Subring.closure s) : f.fieldRange = Subfield.closure s := by
  /-
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    L : Type u_8
    inst✝ : Field L
    g : RingHom A L
    f : RingHom K L
    h : Eq (f.comp (algebraMap A K)) g
    s : Set L
    hs : Eq g.range (Subring.closure s)
    ⊢ Eq f.fieldRange (Subfield.closure s)
  -/
  rw [ringHom_fieldRange_eq_of_comp_eq h, hs]
  /-
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    L : Type u_8
    inst✝ : Field L
    g : RingHom A L
    f : RingHom K L
    h : Eq (f.comp (algebraMap A K)) g
    s : Set L
    hs : Eq g.range (Subring.closure s)
    ⊢ Eq (Subfield.closure ↑(Subring.closure s)) (Subfield.closure s)
  -/
  ext
  /-
    case h
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    L : Type u_8
    inst✝ : Field L
    g : RingHom A L
    f : RingHom K L
    h : Eq (f.comp (algebraMap A K)) g
    s : Set L
    hs : Eq g.range (Subring.closure s)
    x✝ : L
    ⊢ Iff (Membership.mem (Subfield.closure ↑(Subring.closure s)) x✝) (Membership. …
  -/
  simp_rw [Subfield.mem_closure_iff, Subring.closure_eq]
  /-
    🎉 no goals
  -/


/-- Given a commutative ring `A` with field of fractions `K`,
and an injective ring hom `g : A →+* L` where `L` is a field, we get a
field hom sending `z : K` to `g x * (g y)⁻¹`, where `(x, y) : A × (NonZeroDivisors A)` are
such that `z = f x * (f y)⁻¹`. -/
noncomputable def lift (hg : Injective g) : K →+* L :=
  IsLocalization.lift fun y : nonZeroDivisors A => isUnit_map_of_injective hg y


theorem lift_unique (hg : Function.Injective g) {f : K →+* L}
    (hf1 : ∀ x, f (algebraMap A K x) = g x) : IsFractionRing.lift hg = f :=
  IsLocalization.lift_unique _ hf1


/-- Another version of unique to give two lift maps should be equal -/
theorem ringHom_ext {f1 f2 : K →+* L}
    (hf : ∀ x : A, f1 (algebraMap A K x) = f2 (algebraMap A K x)) : f1 = f2 := by
  /-
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : Field K
    L : Type u_7
    inst✝² : Field L
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    f1 f2 : RingHom K L
    hf : ∀ (x : A), Eq (f1 ((algebraMap A K) x)) (f2 ((algebraMap A K) x))
    ⊢ Eq f1 f2
  -/
  ext z
  /-
    case a
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : Field K
    L : Type u_7
    inst✝² : Field L
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    f1 f2 : RingHom K L
    hf : ∀ (x : A), Eq (f1 ((algebraMap A K) x)) (f2 ((algebraMap A K) x))
    z : K
    ⊢ Eq (f1 z) (f2 z)
  -/
  obtain ⟨x, y, hy, rfl⟩ := IsFractionRing.div_surjective (A := A) z
  /-
    case a.intro.intro.intro
    A : Type u_4
    inst✝⁴ : CommRing A
    K : Type u_5
    inst✝³ : Field K
    L : Type u_7
    inst✝² : Field L
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    f1 f2 : RingHom K L
    hf : ∀ (x : A), Eq (f1 ((algebraMap A K) x)) (f2 ((algebraMap A K) x))
    x y : A
    hy : Membership.mem (nonZeroDivisors A) y
    ⊢ Eq (f1 (HDiv.hDiv ((algebraMap A K) x) ((algebraMap A K) y))) (f2 (HDiv.hDiv …
  -/
  rw [map_div₀, map_div₀, hf, hf]
  /-
    🎉 no goals
  -/


theorem injective_comp_algebraMap :
    Function.Injective fun (f : K →+* L) => f.comp (algebraMap A K) :=
  fun _ _ h => ringHom_ext (fun x => RingHom.congr_fun h x)


/-- `AlgHom` version of `IsFractionRing.lift`. -/
noncomputable def liftAlgHom : K →ₐ[R] L :=
  IsLocalization.liftAlgHom fun y : nonZeroDivisors A => isUnit_map_of_injective hg y


theorem liftAlgHom_toRingHom : (liftAlgHom hg : K →ₐ[R] L).toRingHom = lift hg := rfl


@[simp]
theorem coe_liftAlgHom : ⇑(liftAlgHom hg : K →ₐ[R] L) = lift hg := rfl


theorem liftAlgHom_apply : liftAlgHom hg x = lift hg x := rfl


/-- Given a commutative ring `A` with field of fractions `K`,
and an injective ring hom `g : A →+* L` where `L` is a field,
the field hom induced from `K` to `L` maps `x` to `g x` for all
`x : A`. -/
@[simp]
theorem lift_algebraMap (hg : Injective g) (x) : lift hg (algebraMap A K x) = g x :=
  lift_eq _ _


/-- The image of `IsFractionRing.lift` is the subfield generated by the image
of the ring hom. -/
theorem lift_fieldRange (hg : Injective g) :
    (lift hg : K →+* L).fieldRange = Subfield.closure g.range :=
                                       /-
                                         A : Type u_4
                                         inst✝⁴ : CommRing A
                                         K : Type u_5
                                         inst✝³ : Field K
                                         L : Type u_7
                                         inst✝² : Field L
                                         inst✝¹ : Algebra A K
                                         inst✝ : IsFractionRing A K
                                         g : RingHom A L
                                         hg : Function.Injective ⇑g
                                         ⊢ Eq ((IsFractionRing.lift hg).comp (algebraMap A K)) g
                                       -/
  ringHom_fieldRange_eq_of_comp_eq (by ext; simp)
                                            /-
                                              🎉 no goals
                                            -/


/-- The image of `IsFractionRing.lift` is the subfield generated by `s`, if the image
of the ring hom is the subring generated by `s`. -/
theorem lift_fieldRange_eq_of_range_eq (hg : Injective g)
    {s : Set L} (hs : g.range = Subring.closure s) :
    (lift hg : K →+* L).fieldRange = Subfield.closure s :=
                                                   /-
                                                     A : Type u_4
                                                     inst✝⁴ : CommRing A
                                                     K : Type u_5
                                                     inst✝³ : Field K
                                                     L : Type u_7
                                                     inst✝² : Field L
                                                     inst✝¹ : Algebra A K
                                                     inst✝ : IsFractionRing A K
                                                     g : RingHom A L
                                                     hg : Function.Injective ⇑g
                                                     s : Set L
                                                     hs : Eq g.range (Subring.closure s)
                                                     ⊢ Eq ((IsFractionRing.lift hg).comp (algebraMap A K)) g
                                                   -/
  ringHom_fieldRange_eq_of_comp_eq_of_range_eq (by ext; simp) hs
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Given a commutative ring `A` with field of fractions `K`,
and an injective ring hom `g : A →+* L` where `L` is a field,
field hom induced from `K` to `L` maps `f x / f y` to `g x / g y` for all
`x : A, y ∈ NonZeroDivisors A`. -/
theorem lift_mk' (hg : Injective g) (x) (y : nonZeroDivisors A) :
                                          /-
                                            A : Type u_4
                                            inst✝⁴ : CommRing A
                                            K : Type u_5
                                            inst✝³ : Field K
                                            L : Type u_7
                                            inst✝² : Field L
                                            inst✝¹ : Algebra A K
                                            inst✝ : IsFractionRing A K
                                            g : RingHom A L
                                            hg : Function.Injective ⇑g
                                            x : A
                                            y : Subtype fun x => Membership.mem (nonZeroDivisors A) x
                                            ⊢ Eq ((IsFractionRing.lift hg) (IsLocalization.mk' K x y)) (HDiv.hDiv (g x) (g …
                                          -/
    lift hg (mk' K x y) = g x / g y := by simp only [mk'_eq_div, map_div₀, lift_algebraMap]
                                          /-
                                            🎉 no goals
                                          -/


/-- Given commutative rings `A, B` where `B` is an integral domain, with fraction rings `K`, `L`
and an injective ring hom `j : A →+* B`, we get a ring hom
sending `z : K` to `g (j x) * (g (j y))⁻¹`, where `(x, y) : A × (NonZeroDivisors A)` are
such that `z = f x * (f y)⁻¹`. -/
noncomputable def map {A B K L : Type*} [CommRing A] [CommRing B] [IsDomain B] [CommRing K]
    [Algebra A K] [IsFractionRing A K] [CommRing L] [Algebra B L] [IsFractionRing B L] {j : A →+* B}
    (hj : Injective j) : K →+* L :=
  IsLocalization.map L j
    (show nonZeroDivisors A ≤ (nonZeroDivisors B).comap j from
      nonZeroDivisors_le_comap_nonZeroDivisors_of_injective j hj)


/-- Given rings `A, B` and localization maps to their fraction rings
`f : A →+* K, g : B →+* L`, an isomorphism `h : A ≃+* B` induces an isomorphism of
fraction rings `K ≃+* L`. -/
noncomputable def ringEquivOfRingEquiv : K ≃+* L :=
  IsLocalization.ringEquivOfRingEquiv K L h (MulEquivClass.map_nonZeroDivisors h)


@[deprecated (since := "2024-11-05")]
alias fieldEquivOfRingEquiv := ringEquivOfRingEquiv


@[simp]
lemma ringEquivOfRingEquiv_algebraMap
    (a : A) : ringEquivOfRingEquiv h (algebraMap A K a) = algebraMap B L (h a) := by
  /-
    A : Type u_8
    K : Type u_9
    B : Type u_10
    L : Type u_11
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : CommRing K
    inst✝⁴ : CommRing L
    inst✝³ : Algebra A K
    inst✝² : IsFractionRing A K
    inst✝¹ : Algebra B L
    inst✝ : IsFractionRing B L
    h : RingEquiv A B
    a : A
    ⊢ Eq ((IsFractionRing.ringEquivOfRingEquiv h) ((algebraMap A K) a)) ((algebraM …
  -/
  simp [ringEquivOfRingEquiv]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-05")]
alias fieldEquivOfRingEquiv_algebraMap := ringEquivOfRingEquiv_algebraMap


@[simp]
lemma ringEquivOfRingEquiv_symm :
    (ringEquivOfRingEquiv h : K ≃+* L).symm = ringEquivOfRingEquiv h.symm := rfl


/-- Given `R`-algebras `A, B` and localization maps to their fraction rings
`f : A →ₐ[R] K, g : B →ₐ[R] L`, an isomorphism `h : A ≃ₐ[R] B` induces an isomorphism of
fraction rings `K ≃ₐ[R] L`. -/
noncomputable def algEquivOfAlgEquiv : K ≃ₐ[R] L :=
  IsLocalization.algEquivOfAlgEquiv K L h (MulEquivClass.map_nonZeroDivisors h)


@[simp]
lemma algEquivOfAlgEquiv_algebraMap
    (a : A) : algEquivOfAlgEquiv h (algebraMap A K a) = algebraMap B L (h a) := by
  /-
    R : Type u_8
    A : Type u_9
    K : Type u_10
    B : Type u_11
    L : Type u_12
    inst✝¹⁴ : CommSemiring R
    inst✝¹³ : CommRing A
    inst✝¹² : CommRing B
    inst✝¹¹ : CommRing K
    inst✝¹⁰ : CommRing L
    inst✝⁹ : Algebra R A
    inst✝⁸ : Algebra R K
    inst✝⁷ : Algebra A K
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsScalarTower R A K
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra R L
    inst✝² : Algebra B L
    inst✝¹ : IsFractionRing B L
    inst✝ : IsScalarTower R B L
    h : AlgEquiv R A B
    a : A
    ⊢ Eq ((IsFractionRing.algEquivOfAlgEquiv h) ((algebraMap A K) a)) ((algebraMap …
  -/
  simp [algEquivOfAlgEquiv]
  /-
    🎉 no goals
  -/


@[simp]
lemma algEquivOfAlgEquiv_symm :
    (algEquivOfAlgEquiv h : K ≃ₐ[R] L).symm = algEquivOfAlgEquiv h.symm := rfl


/-- An algebra isomorphism of rings induces an algebra isomorphism of fraction fields. -/
noncomputable def fieldEquivOfAlgEquiv (f : B ≃ₐ[A] C) : FB ≃ₐ[FA] FC where
  __ := IsFractionRing.ringEquivOfRingEquiv f.toRingEquiv
  commutes' x := by
    /-
      R : Type u_1
      inst✝⁴¹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁴⁰ : CommRing S
      inst✝³⁹ : Algebra R S
      P : Type u_3
      inst✝³⁸ : CommRing P
      A✝ : Type u_4
      inst✝³⁷ : CommRing A✝
      K : Type u_5
      B✝ : Type u_6
      inst✝³⁶ : CommRing B✝
      inst✝³⁵ : IsDomain B✝
      inst✝³⁴ : Field K
      L : Type u_7
      inst✝³³ : Field L
      inst✝³² : Algebra A✝ K
      inst✝³¹ : IsFractionRing A✝ K
      g : RingHom A✝ L
      A : Type u_8
      B : Type u_9
      C : Type u_10
      D : Type u_11
      inst✝³⁰ : CommRing A
      inst✝²⁹ : CommRing B
      inst✝²⁸ : CommRing C
      inst✝²⁷ : CommRing D
      inst✝²⁶ : Algebra A B
      inst✝²⁵ : Algebra A C
      inst✝²⁴ : Algebra A D
      FA : Type u_12
      FB : Type u_13
      FC : Type u_14
      FD : Type u_15
      inst✝²³ : Field FA
      inst✝²² : Field FB
      inst✝²¹ : Field FC
      inst✝²⁰ : Field FD
      inst✝¹⁹ : Algebra A FA
      inst✝¹⁸ : Algebra B FB
      inst✝¹⁷ : Algebra C FC
      inst✝¹⁶ : Algebra D FD
      inst✝¹⁵ : IsFractionRing A FA
      inst✝¹⁴ : IsFractionRing B FB
      inst✝¹³ : IsFractionRing C FC
      inst✝¹² : IsFractionRing D FD
      inst✝¹¹ : Algebra A FB
      inst✝¹⁰ : IsScalarTower A B FB
      inst✝⁹ : Algebra A FC
      inst✝⁸ : IsScalarTower A C FC
      inst✝⁷ : Algebra A FD
      inst✝⁶ : IsScalarTower A D FD
      inst✝⁵ : Algebra FA FB
      inst✝⁴ : IsScalarTower A FA FB
      inst✝³ : Algebra FA FC
      inst✝² : IsScalarTower A FA FC
      inst✝¹ : Algebra FA FD
      inst✝ : IsScalarTower A FA FD
      f : AlgEquiv A B C
      x : FA
      ⊢ Eq (__spread✝⁻⁰.toFun ((algebraMap FA FB) x)) ((algebraMap FA FC) x)
    -/
    obtain ⟨x, y, -, rfl⟩ := IsFractionRing.div_surjective (A := A) x
    /-
      case intro.intro.intro
      R : Type u_1
      inst✝⁴¹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁴⁰ : CommRing S
      inst✝³⁹ : Algebra R S
      P : Type u_3
      inst✝³⁸ : CommRing P
      A✝ : Type u_4
      inst✝³⁷ : CommRing A✝
      K : Type u_5
      B✝ : Type u_6
      inst✝³⁶ : CommRing B✝
      inst✝³⁵ : IsDomain B✝
      inst✝³⁴ : Field K
      L : Type u_7
      inst✝³³ : Field L
      inst✝³² : Algebra A✝ K
      inst✝³¹ : IsFractionRing A✝ K
      g : RingHom A✝ L
      A : Type u_8
      B : Type u_9
      C : Type u_10
      D : Type u_11
      inst✝³⁰ : CommRing A
      inst✝²⁹ : CommRing B
      inst✝²⁸ : CommRing C
      inst✝²⁷ : CommRing D
      inst✝²⁶ : Algebra A B
      inst✝²⁵ : Algebra A C
      inst✝²⁴ : Algebra A D
      FA : Type u_12
      FB : Type u_13
      FC : Type u_14
      FD : Type u_15
      inst✝²³ : Field FA
      inst✝²² : Field FB
      inst✝²¹ : Field FC
      inst✝²⁰ : Field FD
      inst✝¹⁹ : Algebra A FA
      inst✝¹⁸ : Algebra B FB
      inst✝¹⁷ : Algebra C FC
      inst✝¹⁶ : Algebra D FD
      inst✝¹⁵ : IsFractionRing A FA
      inst✝¹⁴ : IsFractionRing B FB
      inst✝¹³ : IsFractionRing C FC
      inst✝¹² : IsFractionRing D FD
      inst✝¹¹ : Algebra A FB
      inst✝¹⁰ : IsScalarTower A B FB
      inst✝⁹ : Algebra A FC
      inst✝⁸ : IsScalarTower A C FC
      inst✝⁷ : Algebra A FD
      inst✝⁶ : IsScalarTower A D FD
      inst✝⁵ : Algebra FA FB
      inst✝⁴ : IsScalarTower A FA FB
      inst✝³ : Algebra FA FC
      inst✝² : IsScalarTower A FA FC
      inst✝¹ : Algebra FA FD
      inst✝ : IsScalarTower A FA FD
      f : AlgEquiv A B C
      x y : A
      ⊢ Eq (__spread✝⁻⁰.toFun ((algebraMap FA FB) (HDiv.hDiv ((algebraMap A FA) x) ( …
    -/
    simp_rw [map_div₀, ← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply A B FB]
    /-
      case intro.intro.intro
      R : Type u_1
      inst✝⁴¹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁴⁰ : CommRing S
      inst✝³⁹ : Algebra R S
      P : Type u_3
      inst✝³⁸ : CommRing P
      A✝ : Type u_4
      inst✝³⁷ : CommRing A✝
      K : Type u_5
      B✝ : Type u_6
      inst✝³⁶ : CommRing B✝
      inst✝³⁵ : IsDomain B✝
      inst✝³⁴ : Field K
      L : Type u_7
      inst✝³³ : Field L
      inst✝³² : Algebra A✝ K
      inst✝³¹ : IsFractionRing A✝ K
      g : RingHom A✝ L
      A : Type u_8
      B : Type u_9
      C : Type u_10
      D : Type u_11
      inst✝³⁰ : CommRing A
      inst✝²⁹ : CommRing B
      inst✝²⁸ : CommRing C
      inst✝²⁷ : CommRing D
      inst✝²⁶ : Algebra A B
      inst✝²⁵ : Algebra A C
      inst✝²⁴ : Algebra A D
      FA : Type u_12
      FB : Type u_13
      FC : Type u_14
      FD : Type u_15
      inst✝²³ : Field FA
      inst✝²² : Field FB
      inst✝²¹ : Field FC
      inst✝²⁰ : Field FD
      inst✝¹⁹ : Algebra A FA
      inst✝¹⁸ : Algebra B FB
      inst✝¹⁷ : Algebra C FC
      inst✝¹⁶ : Algebra D FD
      inst✝¹⁵ : IsFractionRing A FA
      inst✝¹⁴ : IsFractionRing B FB
      inst✝¹³ : IsFractionRing C FC
      inst✝¹² : IsFractionRing D FD
      inst✝¹¹ : Algebra A FB
      inst✝¹⁰ : IsScalarTower A B FB
      inst✝⁹ : Algebra A FC
      inst✝⁸ : IsScalarTower A C FC
      inst✝⁷ : Algebra A FD
      inst✝⁶ : IsScalarTower A D FD
      inst✝⁵ : Algebra FA FB
      inst✝⁴ : IsScalarTower A FA FB
      inst✝³ : Algebra FA FC
      inst✝² : IsScalarTower A FA FC
      inst✝¹ : Algebra FA FD
      inst✝ : IsScalarTower A FA FD
      f : AlgEquiv A B C
      x y : A
      ⊢ Eq ((IsFractionRing.ringEquivOfRingEquiv f.toRingEquiv).toFun (HDiv.hDiv ((a …
    -/
    simp [← IsScalarTower.algebraMap_apply A C FC]
    /-
      🎉 no goals
    -/


lemma restrictScalars_fieldEquivOfAlgEquiv (f : B ≃ₐ[A] C) :
    (fieldEquivOfAlgEquiv FA FB FC f).restrictScalars A = algEquivOfAlgEquiv f := by
  /-
    A : Type u_8
    B : Type u_9
    C : Type u_10
    inst✝²¹ : CommRing A
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : CommRing C
    inst✝¹⁸ : Algebra A B
    inst✝¹⁷ : Algebra A C
    FA : Type u_12
    FB : Type u_13
    FC : Type u_14
    inst✝¹⁶ : Field FA
    inst✝¹⁵ : Field FB
    inst✝¹⁴ : Field FC
    inst✝¹³ : Algebra A FA
    inst✝¹² : Algebra B FB
    inst✝¹¹ : Algebra C FC
    inst✝¹⁰ : IsFractionRing A FA
    inst✝⁹ : IsFractionRing B FB
    inst✝⁸ : IsFractionRing C FC
    inst✝⁷ : Algebra A FB
    inst✝⁶ : IsScalarTower A B FB
    inst✝⁵ : Algebra A FC
    inst✝⁴ : IsScalarTower A C FC
    inst✝³ : Algebra FA FB
    inst✝² : IsScalarTower A FA FB
    inst✝¹ : Algebra FA FC
    inst✝ : IsScalarTower A FA FC
    f : AlgEquiv A B C
    ⊢ Eq (AlgEquiv.restrictScalars A (IsFractionRing.fieldEquivOfAlgEquiv FA FB FC …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- This says that `fieldEquivOfAlgEquiv f` is an extension of `f` (i.e., it agrees with `f` on
`B`). Whereas `(fieldEquivOfAlgEquiv f).commutes` says that `fieldEquivOfAlgEquiv f` fixes `K`. -/
@[simp]
lemma fieldEquivOfAlgEquiv_algebraMap (f : B ≃ₐ[A] C) (b : B) :
    fieldEquivOfAlgEquiv FA FB FC f (algebraMap B FB b) = algebraMap C FC (f b) :=
  ringEquivOfRingEquiv_algebraMap f.toRingEquiv b


variable (A B) in
@[simp]
lemma fieldEquivOfAlgEquiv_refl :
    fieldEquivOfAlgEquiv FA FB FB (AlgEquiv.refl : B ≃ₐ[A] B) = AlgEquiv.refl := by
  /-
    A : Type u_8
    B : Type u_9
    inst✝¹² : CommRing A
    inst✝¹¹ : CommRing B
    inst✝¹⁰ : Algebra A B
    FA : Type u_12
    FB : Type u_13
    inst✝⁹ : Field FA
    inst✝⁸ : Field FB
    inst✝⁷ : Algebra A FA
    inst✝⁶ : Algebra B FB
    inst✝⁵ : IsFractionRing A FA
    inst✝⁴ : IsFractionRing B FB
    inst✝³ : Algebra A FB
    inst✝² : IsScalarTower A B FB
    inst✝¹ : Algebra FA FB
    inst✝ : IsScalarTower A FA FB
    ⊢ Eq (IsFractionRing.fieldEquivOfAlgEquiv FA FB FB AlgEquiv.refl) AlgEquiv.refl
  -/
  ext x
  /-
    case h
    A : Type u_8
    B : Type u_9
    inst✝¹² : CommRing A
    inst✝¹¹ : CommRing B
    inst✝¹⁰ : Algebra A B
    FA : Type u_12
    FB : Type u_13
    inst✝⁹ : Field FA
    inst✝⁸ : Field FB
    inst✝⁷ : Algebra A FA
    inst✝⁶ : Algebra B FB
    inst✝⁵ : IsFractionRing A FA
    inst✝⁴ : IsFractionRing B FB
    inst✝³ : Algebra A FB
    inst✝² : IsScalarTower A B FB
    inst✝¹ : Algebra FA FB
    inst✝ : IsScalarTower A FA FB
    x : FB
    ⊢ Eq ((IsFractionRing.fieldEquivOfAlgEquiv FA FB FB AlgEquiv.refl) x) (AlgEqui …
  -/
  obtain ⟨x, y, -, rfl⟩ := IsFractionRing.div_surjective (A := B) x
  /-
    case h.intro.intro.intro
    A : Type u_8
    B : Type u_9
    inst✝¹² : CommRing A
    inst✝¹¹ : CommRing B
    inst✝¹⁰ : Algebra A B
    FA : Type u_12
    FB : Type u_13
    inst✝⁹ : Field FA
    inst✝⁸ : Field FB
    inst✝⁷ : Algebra A FA
    inst✝⁶ : Algebra B FB
    inst✝⁵ : IsFractionRing A FA
    inst✝⁴ : IsFractionRing B FB
    inst✝³ : Algebra A FB
    inst✝² : IsScalarTower A B FB
    inst✝¹ : Algebra FA FB
    inst✝ : IsScalarTower A FA FB
    x y : B
    ⊢ Eq ((IsFractionRing.fieldEquivOfAlgEquiv FA FB FB AlgEquiv.refl) (HDiv.hDiv  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma fieldEquivOfAlgEquiv_trans (f : B ≃ₐ[A] C) (g : C ≃ₐ[A] D) :
    fieldEquivOfAlgEquiv FA FB FD (f.trans g) =
      (fieldEquivOfAlgEquiv FA FB FC f).trans (fieldEquivOfAlgEquiv FA FC FD g) := by
  /-
    A : Type u_8
    B : Type u_9
    C : Type u_10
    D : Type u_11
    inst✝³⁰ : CommRing A
    inst✝²⁹ : CommRing B
    inst✝²⁸ : CommRing C
    inst✝²⁷ : CommRing D
    inst✝²⁶ : Algebra A B
    inst✝²⁵ : Algebra A C
    inst✝²⁴ : Algebra A D
    FA : Type u_12
    FB : Type u_13
    FC : Type u_14
    FD : Type u_15
    inst✝²³ : Field FA
    inst✝²² : Field FB
    inst✝²¹ : Field FC
    inst✝²⁰ : Field FD
    inst✝¹⁹ : Algebra A FA
    inst✝¹⁸ : Algebra B FB
    inst✝¹⁷ : Algebra C FC
    inst✝¹⁶ : Algebra D FD
    inst✝¹⁵ : IsFractionRing A FA
    inst✝¹⁴ : IsFractionRing B FB
    inst✝¹³ : IsFractionRing C FC
    inst✝¹² : IsFractionRing D FD
    inst✝¹¹ : Algebra A FB
    inst✝¹⁰ : IsScalarTower A B FB
    inst✝⁹ : Algebra A FC
    inst✝⁸ : IsScalarTower A C FC
    inst✝⁷ : Algebra A FD
    inst✝⁶ : IsScalarTower A D FD
    inst✝⁵ : Algebra FA FB
    inst✝⁴ : IsScalarTower A FA FB
    inst✝³ : Algebra FA FC
    inst✝² : IsScalarTower A FA FC
    inst✝¹ : Algebra FA FD
    inst✝ : IsScalarTower A FA FD
    f : AlgEquiv A B C
    g : AlgEquiv A C D
    ⊢ Eq (IsFractionRing.fieldEquivOfAlgEquiv FA FB FD (f.trans g)) ((IsFractionRi …
  -/
  ext x
  /-
    case h
    A : Type u_8
    B : Type u_9
    C : Type u_10
    D : Type u_11
    inst✝³⁰ : CommRing A
    inst✝²⁹ : CommRing B
    inst✝²⁸ : CommRing C
    inst✝²⁷ : CommRing D
    inst✝²⁶ : Algebra A B
    inst✝²⁵ : Algebra A C
    inst✝²⁴ : Algebra A D
    FA : Type u_12
    FB : Type u_13
    FC : Type u_14
    FD : Type u_15
    inst✝²³ : Field FA
    inst✝²² : Field FB
    inst✝²¹ : Field FC
    inst✝²⁰ : Field FD
    inst✝¹⁹ : Algebra A FA
    inst✝¹⁸ : Algebra B FB
    inst✝¹⁷ : Algebra C FC
    inst✝¹⁶ : Algebra D FD
    inst✝¹⁵ : IsFractionRing A FA
    inst✝¹⁴ : IsFractionRing B FB
    inst✝¹³ : IsFractionRing C FC
    inst✝¹² : IsFractionRing D FD
    inst✝¹¹ : Algebra A FB
    inst✝¹⁰ : IsScalarTower A B FB
    inst✝⁹ : Algebra A FC
    inst✝⁸ : IsScalarTower A C FC
    inst✝⁷ : Algebra A FD
    inst✝⁶ : IsScalarTower A D FD
    inst✝⁵ : Algebra FA FB
    inst✝⁴ : IsScalarTower A FA FB
    inst✝³ : Algebra FA FC
    inst✝² : IsScalarTower A FA FC
    inst✝¹ : Algebra FA FD
    inst✝ : IsScalarTower A FA FD
    f : AlgEquiv A B C
    g : AlgEquiv A C D
    x : FB
    ⊢ Eq ((IsFractionRing.fieldEquivOfAlgEquiv FA FB FD (f.trans g)) x) (((IsFract …
  -/
  obtain ⟨x, y, -, rfl⟩ := IsFractionRing.div_surjective (A := B) x
  /-
    case h.intro.intro.intro
    A : Type u_8
    B : Type u_9
    C : Type u_10
    D : Type u_11
    inst✝³⁰ : CommRing A
    inst✝²⁹ : CommRing B
    inst✝²⁸ : CommRing C
    inst✝²⁷ : CommRing D
    inst✝²⁶ : Algebra A B
    inst✝²⁵ : Algebra A C
    inst✝²⁴ : Algebra A D
    FA : Type u_12
    FB : Type u_13
    FC : Type u_14
    FD : Type u_15
    inst✝²³ : Field FA
    inst✝²² : Field FB
    inst✝²¹ : Field FC
    inst✝²⁰ : Field FD
    inst✝¹⁹ : Algebra A FA
    inst✝¹⁸ : Algebra B FB
    inst✝¹⁷ : Algebra C FC
    inst✝¹⁶ : Algebra D FD
    inst✝¹⁵ : IsFractionRing A FA
    inst✝¹⁴ : IsFractionRing B FB
    inst✝¹³ : IsFractionRing C FC
    inst✝¹² : IsFractionRing D FD
    inst✝¹¹ : Algebra A FB
    inst✝¹⁰ : IsScalarTower A B FB
    inst✝⁹ : Algebra A FC
    inst✝⁸ : IsScalarTower A C FC
    inst✝⁷ : Algebra A FD
    inst✝⁶ : IsScalarTower A D FD
    inst✝⁵ : Algebra FA FB
    inst✝⁴ : IsScalarTower A FA FB
    inst✝³ : Algebra FA FC
    inst✝² : IsScalarTower A FA FC
    inst✝¹ : Algebra FA FD
    inst✝ : IsScalarTower A FA FD
    f : AlgEquiv A B C
    g : AlgEquiv A C D
    x y : B
    ⊢ Eq ((IsFractionRing.fieldEquivOfAlgEquiv FA FB FD (f.trans g)) (HDiv.hDiv (( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An algebra automorphism of a ring induces an algebra automorphism of its fraction field.

This is a bundled version of `fieldEquivOfAlgEquiv`. -/
noncomputable def fieldEquivOfAlgEquivHom : (B ≃ₐ[A] B) →* (L ≃ₐ[K] L) where
  toFun := fieldEquivOfAlgEquiv K L L
  map_one' := fieldEquivOfAlgEquiv_refl A B K L
  map_mul' f g := fieldEquivOfAlgEquiv_trans K L L L g f


@[simp]
lemma fieldEquivOfAlgEquivHom_apply (f : B ≃ₐ[A] B) :
    fieldEquivOfAlgEquivHom K L f = fieldEquivOfAlgEquiv K L L f :=
  rfl


lemma fieldEquivOfAlgEquivHom_injective :
    Function.Injective (fieldEquivOfAlgEquivHom K L : (B ≃ₐ[A] B) →* (L ≃ₐ[K] L)) := by
  /-
    A : Type u_8
    B : Type u_9
    inst✝¹² : CommRing A
    inst✝¹¹ : CommRing B
    inst✝¹⁰ : Algebra A B
    K : Type u_10
    L : Type u_11
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : IsFractionRing A K
    inst✝⁴ : IsFractionRing B L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A B L
    inst✝¹ : Algebra K L
    inst✝ : IsScalarTower A K L
    ⊢ Function.Injective ⇑(IsFractionRing.fieldEquivOfAlgEquivHom K L)
  -/
  intro f g h
  /-
    A : Type u_8
    B : Type u_9
    inst✝¹² : CommRing A
    inst✝¹¹ : CommRing B
    inst✝¹⁰ : Algebra A B
    K : Type u_10
    L : Type u_11
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : IsFractionRing A K
    inst✝⁴ : IsFractionRing B L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A B L
    inst✝¹ : Algebra K L
    inst✝ : IsScalarTower A K L
    f g : AlgEquiv A B B
    h : Eq ((IsFractionRing.fieldEquivOfAlgEquivHom K L) f) ((IsFractionRing.field …
    ⊢ Eq f g
  -/
  ext b
  /-
    case h
    A : Type u_8
    B : Type u_9
    inst✝¹² : CommRing A
    inst✝¹¹ : CommRing B
    inst✝¹⁰ : Algebra A B
    K : Type u_10
    L : Type u_11
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : IsFractionRing A K
    inst✝⁴ : IsFractionRing B L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A B L
    inst✝¹ : Algebra K L
    inst✝ : IsScalarTower A K L
    f g : AlgEquiv A B B
    h : Eq ((IsFractionRing.fieldEquivOfAlgEquivHom K L) f) ((IsFractionRing.field …
    b : B
    ⊢ Eq (f b) (g b)
  -/
  simpa using AlgEquiv.ext_iff.mp h (algebraMap B L b)
  /-
    🎉 no goals
  -/


theorem isFractionRing_iff_of_base_ringEquiv (h : R ≃+* P) :
    IsFractionRing R S ↔
      @IsFractionRing P _ S _ ((algebraMap R S).comp h.symm.toRingHom).toAlgebra := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommRing P
    h : RingEquiv R P
    ⊢ Iff (IsFractionRing R S) (IsFractionRing P S)
  -/
  delta IsFractionRing
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommRing P
    h : RingEquiv R P
    ⊢ Iff (IsLocalization (nonZeroDivisors R) S) (IsLocalization (nonZeroDivisors  …
  -/
  convert isLocalization_iff_of_base_ringEquiv (nonZeroDivisors R) S h
  /-
    case h.e'_2.h.e'_3
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommRing P
    h : RingEquiv R P
    ⊢ Eq (nonZeroDivisors P) (Submonoid.map h.toMonoidHom (nonZeroDivisors R))
  -/
  ext x
  /-
    case h.e'_2.h.e'_3.h
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommRing P
    h : RingEquiv R P
    x : P
    ⊢ Iff (Membership.mem (nonZeroDivisors P) x) (Membership.mem (Submonoid.map h. …
  -/
  erw [Submonoid.map_equiv_eq_comap_symm]
  /-
    case h.e'_2.h.e'_3.h
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommRing P
    h : RingEquiv R P
    x : P
    ⊢ Iff (Membership.mem (nonZeroDivisors P) x) (Membership.mem (Submonoid.comap  …
  -/
  simp only [MulEquiv.coe_toMonoidHom, RingEquiv.toMulEquiv_eq_coe, Submonoid.mem_comap]
  /-
    case h.e'_2.h.e'_3.h
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommRing P
    h : RingEquiv R P
    x : P
    ⊢ Iff (Membership.mem (nonZeroDivisors P) x) (Membership.mem (nonZeroDivisors  …
  -/
  constructor
    /-
      case h.e'_2.h.e'_3.h.mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      ⊢ Membership.mem (nonZeroDivisors P) x → Membership.mem (nonZeroDivisors R) (( …
    -/
  · rintro hx z (hz : z * h.symm x = 0)
    /-
      case h.e'_2.h.e'_3.h.mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      hx : Membership.mem (nonZeroDivisors P) x
      z : R
      hz : Eq (HMul.hMul z (h.symm x)) 0
      ⊢ Eq z 0
    -/
    rw [← h.map_eq_zero_iff]
    /-
      case h.e'_2.h.e'_3.h.mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      hx : Membership.mem (nonZeroDivisors P) x
      z : R
      hz : Eq (HMul.hMul z (h.symm x)) 0
      ⊢ Eq (h z) 0
    -/
    apply hx
    /-
      case h.e'_2.h.e'_3.h.mp.a
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      hx : Membership.mem (nonZeroDivisors P) x
      z : R
      hz : Eq (HMul.hMul z (h.symm x)) 0
      ⊢ Eq (HMul.hMul (h z) x) 0
    -/
    simpa only [h.map_zero, h.apply_symm_apply, h.map_mul] using congr_arg h hz
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_3.h.mpr
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      ⊢ Membership.mem (nonZeroDivisors R) ((↑h).symm x) → Membership.mem (nonZeroDi …
    -/
  · rintro (hx : h.symm x ∈ _) z hz
    /-
      case h.e'_2.h.e'_3.h.mpr
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      hx : Membership.mem (nonZeroDivisors R) (h.symm x)
      z : P
      hz : Eq (HMul.hMul z x) 0
      ⊢ Eq z 0
    -/
    rw [← h.symm.map_eq_zero_iff]
    /-
      case h.e'_2.h.e'_3.h.mpr
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      hx : Membership.mem (nonZeroDivisors R) (h.symm x)
      z : P
      hz : Eq (HMul.hMul z x) 0
      ⊢ Eq (h.symm z) 0
    -/
    apply hx
    /-
      case h.e'_2.h.e'_3.h.mpr.a
      R : Type u_1
      inst✝³ : CommRing R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommRing P
      h : RingEquiv R P
      x : P
      hx : Membership.mem (nonZeroDivisors R) (h.symm x)
      z : P
      hz : Eq (HMul.hMul z x) 0
      ⊢ Eq (HMul.hMul (h.symm z) (h.symm x)) 0
    -/
    rw [← h.symm.map_mul, hz, h.symm.map_zero]
    /-
      🎉 no goals
    -/


protected theorem nontrivial (R S : Type*) [CommRing R] [Nontrivial R] [CommRing S] [Algebra R S]
    [IsFractionRing R S] : Nontrivial S := by
  /-
    R : Type u_8
    S : Type u_9
    inst✝⁴ : CommRing R
    inst✝³ : Nontrivial R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsFractionRing R S
    ⊢ Nontrivial S
  -/
  apply nontrivial_of_ne
    /-
      case h
      R : Type u_8
      S : Type u_9
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsFractionRing R S
      ⊢ Ne ?x ?y
    -/
  · intro h
    /-
      case h
      R : Type u_8
      S : Type u_9
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsFractionRing R S
      h : Eq ?x ?y
      ⊢ False
    -/
    apply @zero_ne_one R
    exact
      IsLocalization.injective S (le_of_eq rfl)
        (((algebraMap R S).map_zero.trans h).trans (algebraMap R S).map_one.symm)


theorem algebraMap_injective_of_field_isFractionRing (K L : Type*) [Field K] [Semiring L]
    [Nontrivial L] [Algebra R K] [IsFractionRing R K] [Algebra S L] [Algebra K L] [Algebra R L]
    [IsScalarTower R S L] [IsScalarTower R K L] : Function.Injective (algebraMap R S) := by
  /-
    R : Type u_1
    inst✝¹² : CommRing R
    S : Type u_2
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra R S
    K : Type u_6
    L : Type u_7
    inst✝⁹ : Field K
    inst✝⁸ : Semiring L
    inst✝⁷ : Nontrivial L
    inst✝⁶ : Algebra R K
    inst✝⁵ : IsFractionRing R K
    inst✝⁴ : Algebra S L
    inst✝³ : Algebra K L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    ⊢ Function.Injective ⇑(algebraMap R S)
  -/
  refine Function.Injective.of_comp (f := algebraMap S L) ?_
  /-
    R : Type u_1
    inst✝¹² : CommRing R
    S : Type u_2
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra R S
    K : Type u_6
    L : Type u_7
    inst✝⁹ : Field K
    inst✝⁸ : Semiring L
    inst✝⁷ : Nontrivial L
    inst✝⁶ : Algebra R K
    inst✝⁵ : IsFractionRing R K
    inst✝⁴ : Algebra S L
    inst✝³ : Algebra K L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    ⊢ Function.Injective (Function.comp ⇑(algebraMap S L) ⇑(algebraMap R S))
  -/
  rw [← RingHom.coe_comp, ← IsScalarTower.algebraMap_eq, IsScalarTower.algebraMap_eq R K L]
  /-
    R : Type u_1
    inst✝¹² : CommRing R
    S : Type u_2
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra R S
    K : Type u_6
    L : Type u_7
    inst✝⁹ : Field K
    inst✝⁸ : Semiring L
    inst✝⁷ : Nontrivial L
    inst✝⁶ : Algebra R K
    inst✝⁵ : IsFractionRing R K
    inst✝⁴ : Algebra S L
    inst✝³ : Algebra K L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    ⊢ Function.Injective ⇑((algebraMap K L).comp (algebraMap R K))
  -/
  exact (algebraMap K L).injective.comp (IsFractionRing.injective R K)
  /-
    🎉 no goals
  -/


theorem NoZeroSMulDivisors.of_field_isFractionRing [NoZeroDivisors S] (K L : Type*) [Field K]
    [Semiring L] [Nontrivial L] [Algebra R K] [IsFractionRing R K] [Algebra S L] [Algebra K L]
    [Algebra R L] [IsScalarTower R S L] [IsScalarTower R K L] : NoZeroSMulDivisors R S :=
  of_algebraMap_injective (algebraMap_injective_of_field_isFractionRing R S K L)


/-- The fraction ring of a commutative ring `R` as a quotient type.

We instantiate this definition as generally as possible, and assume that the
commutative ring `R` is an integral domain only when this is needed for proving.

In this generality, this construction is also known as the *total fraction ring* of `R`.
-/
abbrev FractionRing :=
  Localization (nonZeroDivisors R)


instance unique [Subsingleton R] : Unique (FractionRing R) := inferInstance


instance [Nontrivial R] : Nontrivial (FractionRing R) := inferInstance


/-- Porting note: if the fields of this instance are explicitly defined as they were
in mathlib3, the last instance in this file suffers a TC timeout -/
noncomputable instance field : Field (FractionRing A) := inferInstance


@[simp]
theorem mk_eq_div {r s} :
    (Localization.mk r s : FractionRing A) =
      (algebraMap _ _ r / algebraMap A _ s : FractionRing A) := by
  /-
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    r : A
    s : Subtype fun x => Membership.mem (nonZeroDivisors A) x
    ⊢ Eq (Localization.mk r s) (HDiv.hDiv ((algebraMap A (FractionRing A)) r) ((al …
  -/
  rw [Localization.mk_eq_mk', IsFractionRing.mk'_eq_div]
  /-
    🎉 no goals
  -/


/-- This is not an instance because it creates a diamond when `K = FractionRing R`.
Should usually be introduced locally along with `isScalarTower_liftAlgebra`
See note [reducible non-instances]. -/
noncomputable abbrev liftAlgebra [IsDomain R] [Field K] [Algebra R K]
    [NoZeroSMulDivisors R K] : Algebra (FractionRing R) K :=
  RingHom.toAlgebra (IsFractionRing.lift (NoZeroSMulDivisors.algebraMap_injective R _))

-- Porting note: had to fill in the `_` by hand for this instance

instance isScalarTower_liftAlgebra [IsDomain R] [Field K] [Algebra R K] [NoZeroSMulDivisors R K] :
       /-
         R : Type u_1
         inst✝⁹ : CommRing R
         M : Submonoid R
         S : Type u_2
         inst✝⁸ : CommRing S
         inst✝⁷ : Algebra R S
         P : Type u_3
         inst✝⁶ : CommRing P
         A : Type u_4
         inst✝⁵ : CommRing A
         K : Type u_5
         inst✝⁴ : IsDomain A
         inst✝³ : IsDomain R
         inst✝² : Field K
         inst✝¹ : Algebra R K
         inst✝ : NoZeroSMulDivisors R K
         ⊢ Sort ?u.260354
       -/
    by letI := liftAlgebra R K; exact IsScalarTower R (FractionRing R) K := by
                                /-
                                  🎉 no goals
                                -/
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Type u_3
    inst✝⁶ : CommRing P
    A : Type u_4
    inst✝⁵ : CommRing A
    K : Type u_5
    inst✝⁴ : IsDomain A
    inst✝³ : IsDomain R
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : NoZeroSMulDivisors R K
    ⊢ IsScalarTower R (FractionRing R) K
  -/
  letI := liftAlgebra R K
  exact IsScalarTower.of_algebraMap_eq fun x =>
    (IsFractionRing.lift_algebraMap (NoZeroSMulDivisors.algebraMap_injective R K) x).symm


/-- Given a ring `A` and a localization map to a fraction ring
`f : A →+* K`, we get an `A`-isomorphism between the fraction ring of `A` as a quotient
type and `K`. -/
noncomputable def algEquiv (K : Type*) [CommRing K] [Algebra A K] [IsFractionRing A K] :
    FractionRing A ≃ₐ[A] K :=
  Localization.algEquiv (nonZeroDivisors A) K


instance [Algebra R A] [NoZeroSMulDivisors R A] : NoZeroSMulDivisors R (FractionRing A) := by
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    P : Type u_3
    inst✝⁴ : CommRing P
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ NoZeroSMulDivisors R (FractionRing A)
  -/
  apply NoZeroSMulDivisors.of_algebraMap_injective
  /-
    case h
    R : Type u_1
    inst✝⁷ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    P : Type u_3
    inst✝⁴ : CommRing P
    A : Type u_4
    inst✝³ : CommRing A
    K : Type u_5
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ Function.Injective ⇑(algebraMap R (FractionRing A))
  -/
  rw [IsScalarTower.algebraMap_eq R A]
  apply Function.Injective.comp (NoZeroSMulDivisors.algebraMap_injective A (FractionRing A))
    (NoZeroSMulDivisors.algebraMap_injective R A)


