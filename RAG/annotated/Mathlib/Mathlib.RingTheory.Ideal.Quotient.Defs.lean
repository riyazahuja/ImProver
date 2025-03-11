/-- The quotient `R/I` of a ring `R` by an ideal `I`.

The ideal quotient of `I` is defined to equal the quotient of `I` as an `R`-submodule of `R`.
This definition uses `abbrev` so that typeclass instances can be shared between
`Ideal.Quotient I` and `Submodule.Quotient I`.
-/
@[instance] abbrev instHasQuotient : HasQuotient R (Ideal R) :=
  Submodule.hasQuotient


instance one (I : Ideal R) : One (R ⧸ I) :=
  ⟨Submodule.Quotient.mk 1⟩


/-- On `Ideal`s, `Submodule.quotientRel` is a ring congruence. -/
protected def ringCon (I : Ideal R) : RingCon R :=
  { QuotientAddGroup.con I.toAddSubgroup with
    mul' := fun {a₁ b₁ a₂ b₂} h₁ h₂ => by
      /-
        R : Type u
        inst✝ : CommRing R
        I✝ : Ideal R
        a b : R
        S : Type v
        x y : R
        I : Ideal R
        a₁ b₁ a₂ b₂ : R
        h₁ : __src✝.toSetoid a₁ b₁
        h₂ : __src✝.toSetoid a₂ b₂
        ⊢ __src✝.toSetoid (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂)
      -/
      rw [Submodule.quotientRel_def] at h₁ h₂ ⊢
      /-
        R : Type u
        inst✝ : CommRing R
        I✝ : Ideal R
        a b : R
        S : Type v
        x y : R
        I : Ideal R
        a₁ b₁ a₂ b₂ : R
        h₁ : Membership.mem I (HSub.hSub a₁ b₁)
        h₂ : Membership.mem I (HSub.hSub a₂ b₂)
        ⊢ Membership.mem I (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))
      -/
      have F := I.add_mem (I.mul_mem_left a₂ h₁) (I.mul_mem_right b₁ h₂)
      have : a₁ * a₂ - b₁ * b₂ = a₂ * (a₁ - b₁) + (a₂ - b₂) * b₁ := by
        rw [mul_sub, sub_mul, sub_add_sub_cancel, mul_comm, mul_comm b₁]
      /-
        R : Type u
        inst✝ : CommRing R
        I✝ : Ideal R
        a b : R
        S : Type v
        x y : R
        I : Ideal R
        a₁ b₁ a₂ b₂ : R
        h₁ : Membership.mem I (HSub.hSub a₁ b₁)
        h₂ : Membership.mem I (HSub.hSub a₂ b₂)
        F : Membership.mem I (HAdd.hAdd (HMul.hMul a₂ (HSub.hSub a₁ b₁)) (HMul.hMul (H …
        this : Eq (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂)) (HAdd.hAdd (HMul.hMu …
        ⊢ Membership.mem I (HSub.hSub (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))
      -/
      rwa [← this] at F }
      /-
        🎉 no goals
      -/

-- This instance makes use of the existing AddCommGroup instance to boost performance.

instance commRing (I : Ideal R) : CommRing (R ⧸ I) where
  __ : AddCommGroup (R ⧸ I) := inferInstance
  __ : CommRing (Quotient.ringCon I).Quotient := inferInstance

-- Sanity test to make sure no diamonds have emerged in `commRing`

/-- The ring homomorphism from a ring `R` to a quotient ring `R/I`. -/
def mk (I : Ideal R) : R →+* R ⧸ I where
  toFun a := Submodule.Quotient.mk a
  map_zero' := rfl
  map_one' := rfl
  map_mul' _ _ := rfl
  map_add' _ _ := rfl


instance {I : Ideal R} : Coe R (R ⧸ I) :=
  ⟨Ideal.Quotient.mk I⟩


/-- Two `RingHom`s from the quotient by an ideal are equal if their
compositions with `Ideal.Quotient.mk'` are equal.

See note [partially-applied ext lemmas]. -/
@[ext 1100]
theorem ringHom_ext [NonAssocSemiring S] ⦃f g : R ⧸ I →+* S⦄ (h : f.comp (mk I) = g.comp (mk I)) :
    f = g :=
  RingHom.ext fun x => Quotient.inductionOn' x <| (RingHom.congr_fun h : _)


instance inhabited : Inhabited (R ⧸ I) :=
  ⟨mk I 37⟩


protected theorem eq : mk I x = mk I y ↔ x - y ∈ I :=
  Submodule.Quotient.eq I


@[simp]
theorem mk_eq_mk (x : R) : (Submodule.Quotient.mk x : R ⧸ I) = mk I x := rfl


theorem eq_zero_iff_mem {I : Ideal R} : mk I a = 0 ↔ a ∈ I :=
  Submodule.Quotient.mk_eq_zero _


theorem mk_eq_mk_iff_sub_mem (x y : R) : mk I x = mk I y ↔ x - y ∈ I := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x y : R
    ⊢ Iff (Eq ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)) (Membership.mem …
  -/
  rw [← eq_zero_iff_mem, map_sub, sub_eq_zero]
  /-
    🎉 no goals
  -/


theorem mk_surjective : Function.Surjective (mk I) := fun y =>
  Quotient.inductionOn' y fun x => Exists.intro x rfl


instance : RingHomSurjective (mk I) :=
  ⟨mk_surjective⟩


/-- If `I` is an ideal of a commutative ring `R`, if `q : R → R/I` is the quotient map, and if
`s ⊆ R` is a subset, then `q⁻¹(q(s)) = ⋃ᵢ(i + s)`, the union running over all `i ∈ I`. -/
theorem quotient_ring_saturate (I : Ideal R) (s : Set R) :
    mk I ⁻¹' (mk I '' s) = ⋃ x : I, (fun y => x.1 + y) '' s := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    s : Set R
    ⊢ Eq (Set.preimage (⇑(Ideal.Quotient.mk I)) (Set.image (⇑(Ideal.Quotient.mk I) …
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    s : Set R
    x : R
    ⊢ Iff (Membership.mem (Set.preimage (⇑(Ideal.Quotient.mk I)) (Set.image (⇑(Ide …
  -/
  simp only [mem_preimage, mem_image, mem_iUnion, Ideal.Quotient.eq]
  exact
    ⟨fun ⟨a, a_in, h⟩ => ⟨⟨_, I.neg_mem h⟩, a, a_in, by simp⟩, fun ⟨⟨i, hi⟩, a, ha, Eq⟩ =>
      ⟨a, ha, by rw [← Eq, sub_add_eq_sub_sub_swap, sub_self, zero_sub]; exact I.neg_mem hi⟩⟩


/-- Given a ring homomorphism `f : R →+* S` sending all elements of an ideal to zero,
lift it to the quotient by this ideal. -/
def lift (I : Ideal R) (f : R →+* S) (H : ∀ a : R, a ∈ I → f a = 0) : R ⧸ I →+* S :=
  { QuotientAddGroup.lift I.toAddSubgroup f.toAddMonoidHom H with
    map_one' := f.map_one
    map_mul' := fun a₁ a₂ => Quotient.inductionOn₂' a₁ a₂ f.map_mul }


@[simp]
theorem lift_mk (I : Ideal R) (f : R →+* S) (H : ∀ a : R, a ∈ I → f a = 0) :
    lift I f H (mk I a) = f a :=
  rfl


theorem lift_surjective_of_surjective (I : Ideal R) {f : R →+* S} (H : ∀ a : R, a ∈ I → f a = 0)
    (hf : Function.Surjective f) : Function.Surjective (Ideal.Quotient.lift I f H) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hf : Function.Surjective ⇑f
    ⊢ Function.Surjective ⇑(Ideal.Quotient.lift I f H)
  -/
  intro y
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hf : Function.Surjective ⇑f
    y : S
    ⊢ Exists fun a => Eq ((Ideal.Quotient.lift I f H) a) y
  -/
  obtain ⟨x, rfl⟩ := hf y
  /-
    case intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hf : Function.Surjective ⇑f
    x : R
    ⊢ Exists fun a => Eq ((Ideal.Quotient.lift I f H) a) (f x)
  -/
  use Ideal.Quotient.mk I x
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hf : Function.Surjective ⇑f
    x : R
    ⊢ Eq ((Ideal.Quotient.lift I f H) ((Ideal.Quotient.mk I) x)) (f x)
  -/
  simp only [Ideal.Quotient.lift_mk]
  /-
    🎉 no goals
  -/


/-- The ring homomorphism from the quotient by a smaller ideal to the quotient by a larger ideal.

This is the `Ideal.Quotient` version of `Quot.Factor` -/
def factor (S T : Ideal R) (H : S ≤ T) : R ⧸ S →+* R ⧸ T :=
  Ideal.Quotient.lift S (mk T) fun _ hx => eq_zero_iff_mem.2 (H hx)


@[simp]
theorem factor_mk (S T : Ideal R) (H : S ≤ T) (x : R) : factor S T H (mk S x) = mk T x :=
  rfl


@[simp]
theorem factor_comp_mk (S T : Ideal R) (H : S ≤ T) : (factor S T H).comp (mk S) = mk T := by
  /-
    R : Type u
    inst✝ : CommRing R
    S T : Ideal R
    H : LE.le S T
    ⊢ Eq ((Ideal.Quotient.factor S T H).comp (Ideal.Quotient.mk S)) (Ideal.Quotien …
  -/
  ext x
  /-
    case a
    R : Type u
    inst✝ : CommRing R
    S T : Ideal R
    H : LE.le S T
    x : R
    ⊢ Eq (((Ideal.Quotient.factor S T H).comp (Ideal.Quotient.mk S)) x) ((Ideal.Qu …
  -/
  rw [RingHom.comp_apply, factor_mk]
  /-
    🎉 no goals
  -/


/-- Quotienting by equal ideals gives equivalent rings.

See also `Submodule.quotEquivOfEq` and `Ideal.quotientEquivAlgOfEq`.
-/
def quotEquivOfEq {R : Type*} [CommRing R] {I J : Ideal R} (h : I = J) : R ⧸ I ≃+* R ⧸ J :=
  { Submodule.quotEquivOfEq I J h with
    map_mul' := by
      /-
        R✝ : Type u
        inst✝¹ : CommRing R✝
        I✝ : Ideal R✝
        a b : R✝
        S : Type v
        R : Type u_1
        inst✝ : CommRing R
        I J : Ideal R
        h : Eq I J
        ⊢ ∀ (x y : HasQuotient.Quotient R I), Eq ({ toFun := (↑__src✝).toFun, invFun : …
      -/
      rintro ⟨x⟩ ⟨y⟩
      /-
        case mk.mk
        R✝ : Type u
        inst✝¹ : CommRing R✝
        I✝ : Ideal R✝
        a b : R✝
        S : Type v
        R : Type u_1
        inst✝ : CommRing R
        I J : Ideal R
        h : Eq I J
        x✝ : HasQuotient.Quotient R I
        x : R
        y✝ : HasQuotient.Quotient R I
        y : R
        ⊢ Eq ({ toFun := (↑__src✝).toFun, invFun := __src✝.invFun, left_inv := ⋯, righ …
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[simp]
theorem quotEquivOfEq_mk {R : Type*} [CommRing R] {I J : Ideal R} (h : I = J) (x : R) :
    quotEquivOfEq h (Ideal.Quotient.mk I x) = Ideal.Quotient.mk J x :=
  rfl


@[simp]
theorem quotEquivOfEq_symm {R : Type*} [CommRing R] {I J : Ideal R} (h : I = J) :
                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝ : CommRing R
                                                                      I J : Ideal R
                                                                      h : Eq I J
                                                                      ⊢ Eq (Ideal.quotEquivOfEq h).symm (Ideal.quotEquivOfEq ⋯)
                                                                    -/
    (Ideal.quotEquivOfEq h).symm = Ideal.quotEquivOfEq h.symm := by ext; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


