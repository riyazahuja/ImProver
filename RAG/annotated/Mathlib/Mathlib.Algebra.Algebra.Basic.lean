instance _root_.PUnit.algebra : Algebra R PUnit.{v + 1} where
  toFun _ := PUnit.unit
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl
  commutes' _ _ := rfl
  smul_def' _ _ := rfl


@[simp]
theorem algebraMap_pUnit (r : R) : algebraMap R PUnit r = PUnit.unit :=
  rfl


instance _root_.ULift.algebra : Algebra R (ULift A) :=
  { ULift.module',
    (ULift.ringEquiv : ULift A ≃+* A).symm.toRingHom.comp (algebraMap R A) with
    toFun := fun r => ULift.up (algebraMap R A r)
    commutes' := fun r x => ULift.down_injective <| Algebra.commutes r x.down
    smul_def' := fun r x => ULift.down_injective <| Algebra.smul_def' r x.down }


theorem _root_.ULift.algebraMap_eq (r : R) :
    algebraMap R (ULift A) r = ULift.up (algebraMap R A r) :=
  rfl


@[simp]
theorem _root_.ULift.down_algebraMap (r : R) : (algebraMap R (ULift A) r).down = algebraMap R A r :=
  rfl


/-- Algebra over a subsemiring. This builds upon `Subsemiring.module`. -/
instance ofSubsemiring (S : Subsemiring R) : Algebra S A where
  toRingHom := (algebraMap R A).comp S.subtype
  smul := (· • ·)
  commutes' r x := Algebra.commutes (r : R) x
  smul_def' r x := Algebra.smul_def (r : R) x


theorem algebraMap_ofSubsemiring (S : Subsemiring R) :
    (algebraMap S R : S →+* R) = Subsemiring.subtype S :=
  rfl


theorem coe_algebraMap_ofSubsemiring (S : Subsemiring R) : (algebraMap S R : S → R) = Subtype.val :=
  rfl


theorem algebraMap_ofSubsemiring_apply (S : Subsemiring R) (x : S) : algebraMap S R x = x :=
  rfl


/-- Algebra over a subring. This builds upon `Subring.module`. -/
instance ofSubring {R A : Type*} [CommRing R] [Ring A] [Algebra R A] (S : Subring R) :
    Algebra S A where -- Porting note: don't use `toSubsemiring` because of a timeout
  toRingHom := (algebraMap R A).comp S.subtype
  smul := (· • ·)
  commutes' r x := Algebra.commutes (r : R) x
  smul_def' r x := Algebra.smul_def (r : R) x


theorem algebraMap_ofSubring {R : Type*} [CommRing R] (S : Subring R) :
    (algebraMap S R : S →+* R) = Subring.subtype S :=
  rfl


theorem coe_algebraMap_ofSubring {R : Type*} [CommRing R] (S : Subring R) :
    (algebraMap S R : S → R) = Subtype.val :=
  rfl


theorem algebraMap_ofSubring_apply {R : Type*} [CommRing R] (S : Subring R) (x : S) :
    algebraMap S R x = x :=
  rfl


/-- Explicit characterization of the submonoid map in the case of an algebra.
`S` is made explicit to help with type inference -/
def algebraMapSubmonoid (S : Type*) [Semiring S] [Algebra R S] (M : Submonoid R) : Submonoid S :=
  M.map (algebraMap R S)


theorem mem_algebraMapSubmonoid_of_mem {S : Type*} [Semiring S] [Algebra R S] {M : Submonoid R}
    (x : M) : algebraMap R S x ∈ algebraMapSubmonoid S M :=
  Set.mem_image_of_mem (algebraMap R S) x.2


theorem mul_sub_algebraMap_commutes [Ring A] [Algebra R A] (x : A) (r : R) :
                                                                  /-
                                                                    R : Type u
                                                                    A : Type w
                                                                    inst✝² : CommSemiring R
                                                                    inst✝¹ : Ring A
                                                                    inst✝ : Algebra R A
                                                                    x : A
                                                                    r : R
                                                                    ⊢ Eq (HMul.hMul x (HSub.hSub x ((algebraMap R A) r))) (HMul.hMul (HSub.hSub x  …
                                                                  -/
    x * (x - algebraMap R A r) = (x - algebraMap R A r) * x := by rw [mul_sub, ← commutes, sub_mul]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem mul_sub_algebraMap_pow_commutes [Ring A] [Algebra R A] (x : A) (r : R) (n : ℕ) :
    x * (x - algebraMap R A r) ^ n = (x - algebraMap R A r) ^ n * x := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ', ← mul_assoc, mul_sub_algebraMap_commutes, mul_assoc, ih, ← mul_assoc]


/-- A `Semiring` that is an `Algebra` over a commutative ring carries a natural `Ring` structure.
See note [reducible non-instances]. -/
abbrev semiringToRing (R : Type*) [CommRing R] [Semiring A] [Algebra R A] : Ring A :=
  { __ := (inferInstance : Semiring A)
    __ := Module.addCommMonoidToAddCommGroup R
    intCast := fun z => algebraMap R A z
                                 /-
                                   R✝ : Type u
                                   A : Type w
                                   R : Type u_1
                                   inst✝² : CommRing R
                                   inst✝¹ : Semiring A
                                   inst✝ : Algebra R A
                                   z : Nat
                                   ⊢ Eq (IntCast.intCast ↑z) ↑z
                                 -/
    intCast_ofNat := fun z => by simp only [Int.cast_natCast, map_natCast]
                                 /-
                                   🎉 no goals
                                 -/
                                   /-
                                     R✝ : Type u
                                     A : Type w
                                     R : Type u_1
                                     inst✝² : CommRing R
                                     inst✝¹ : Semiring A
                                     inst✝ : Algebra R A
                                     z : Nat
                                     ⊢ Eq (IntCast.intCast (Int.negSucc z)) (Neg.neg ↑(HAdd.hAdd z 1))
                                   -/
    intCast_negSucc := fun z => by simp }
                                   /-
                                     🎉 no goals
                                   -/


instance {R : Type*} [Ring R] : Algebra (Subring.center R) R where
  toFun := Subtype.val
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl
  commutes' r x := (Subring.mem_center_iff.1 r.2 x).symm
  smul_def' _ _ := rfl


instance End.instAlgebra : Algebra R (Module.End S M) :=
  Algebra.ofModule smul_mul_assoc fun r f g => (smul_comm r f g).symm

-- to prove this is a special case of the above

theorem algebraMap_end_eq_smul_id (a : R) : algebraMap R (End S M) a = a • LinearMap.id :=
  rfl


@[simp]
theorem algebraMap_end_apply (a : R) (m : M) : algebraMap R (End S M) a m = a • m :=
  rfl


@[simp]
theorem ker_algebraMap_end (K : Type u) (V : Type v) [Field K] [AddCommGroup V] [Module K V] (a : K)
    (ha : a ≠ 0) : LinearMap.ker ((algebraMap K (End K V)) a) = ⊥ :=
  LinearMap.ker_smul _ _ ha


theorem End_algebraMap_isUnit_inv_apply_eq_iff {x : R}
    (h : IsUnit (algebraMap R (Module.End S M) x)) (m m' : M) :
    (↑(h.unit⁻¹) : Module.End S M) m = m' ↔ m = x • m' where
  mp H := H ▸ (End_isUnit_apply_inv_apply_of_isUnit h m).symm
  mpr H :=
    H.symm ▸ by
      /-
        R : Type u
        S : Type v
        M : Type w
        inst✝⁷ : CommSemiring R
        inst✝⁶ : Semiring S
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : SMulCommClass S R M
        inst✝¹ : SMul R S
        inst✝ : IsScalarTower R S M
        x : R
        h : IsUnit ((algebraMap R (Module.End S M)) x)
        m m' : M
        H : Eq m (HSMul.hSMul x m')
        ⊢ Eq (↑(Inv.inv h.unit) (HSMul.hSMul x m')) m'
      -/
      apply_fun ⇑h.unit.val using ((Module.End_isUnit_iff _).mp h).injective
      /-
        R : Type u
        S : Type v
        M : Type w
        inst✝⁷ : CommSemiring R
        inst✝⁶ : Semiring S
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : SMulCommClass S R M
        inst✝¹ : SMul R S
        inst✝ : IsScalarTower R S M
        x : R
        h : IsUnit ((algebraMap R (Module.End S M)) x)
        m m' : M
        H : Eq m (HSMul.hSMul x m')
        ⊢ Eq (↑h.unit (↑(Inv.inv h.unit) (HSMul.hSMul x m'))) (↑h.unit m')
      -/
      simpa using End_isUnit_apply_inv_apply_of_isUnit h (x • m')
      /-
        🎉 no goals
      -/


theorem End_algebraMap_isUnit_inv_apply_eq_iff' {x : R}
    (h : IsUnit (algebraMap R (Module.End S M) x)) (m m' : M) :
    m' = (↑h.unit⁻¹ : Module.End S M) m ↔ m = x • m' where
  mp H := H ▸ (End_isUnit_apply_inv_apply_of_isUnit h m).symm
  mpr H :=
    H.symm ▸ by
      /-
        R : Type u
        S : Type v
        M : Type w
        inst✝⁷ : CommSemiring R
        inst✝⁶ : Semiring S
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : SMulCommClass S R M
        inst✝¹ : SMul R S
        inst✝ : IsScalarTower R S M
        x : R
        h : IsUnit ((algebraMap R (Module.End S M)) x)
        m m' : M
        H : Eq m (HSMul.hSMul x m')
        ⊢ Eq m' (↑(Inv.inv h.unit) (HSMul.hSMul x m'))
      -/
      apply_fun (↑h.unit : M → M) using ((Module.End_isUnit_iff _).mp h).injective
      /-
        R : Type u
        S : Type v
        M : Type w
        inst✝⁷ : CommSemiring R
        inst✝⁶ : Semiring S
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : SMulCommClass S R M
        inst✝¹ : SMul R S
        inst✝ : IsScalarTower R S M
        x : R
        h : IsUnit ((algebraMap R (Module.End S M)) x)
        m m' : M
        H : Eq m (HSMul.hSMul x m')
        ⊢ Eq (↑h.unit m') (↑h.unit (↑(Inv.inv h.unit) (HSMul.hSMul x m')))
      -/
      simpa using End_isUnit_apply_inv_apply_of_isUnit h (x • m') |>.symm
      /-
        🎉 no goals
      -/


/-- An alternate statement of `LinearMap.map_smul` for when `algebraMap` is more convenient to
work with than `•`. -/
theorem map_algebraMap_mul (f : A →ₗ[R] B) (a : A) (r : R) :
    f (algebraMap R A r * a) = algebraMap R B r * f a := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : LinearMap (RingHom.id R) A B
    a : A
    r : R
    ⊢ Eq (f (HMul.hMul ((algebraMap R A) r) a)) (HMul.hMul ((algebraMap R B) r) (f …
  -/
  rw [← Algebra.smul_def, ← Algebra.smul_def, map_smul]
  /-
    🎉 no goals
  -/


theorem map_mul_algebraMap (f : A →ₗ[R] B) (a : A) (r : R) :
    f (a * algebraMap R A r) = f a * algebraMap R B r := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : LinearMap (RingHom.id R) A B
    a : A
    r : R
    ⊢ Eq (f (HMul.hMul a ((algebraMap R A) r))) (HMul.hMul (f a) ((algebraMap R B) …
  -/
  rw [← Algebra.commutes, ← Algebra.commutes, map_algebraMap_mul]
  /-
    🎉 no goals
  -/


/-- Semiring ⥤ ℕ-Alg -/
instance (priority := 99) Semiring.toNatAlgebra : Algebra ℕ R where
  commutes' := Nat.cast_commute
  smul_def' _ _ := nsmul_eq_mul _ _
  toRingHom := Nat.castRingHom R


instance nat_algebra_subsingleton : Subsingleton (Algebra ℕ R) :=
                 /-
                   R : Type u_1
                   inst✝ : Semiring R
                   P Q : Algebra Nat R
                   ⊢ Eq P Q
                 -/
  ⟨fun P Q => by ext; simp⟩
                      /-
                        🎉 no goals
                      -/


/-- Ring ⥤ ℤ-Alg -/
instance (priority := 99) Ring.toIntAlgebra : Algebra ℤ R where
  commutes' := Int.cast_commute
  smul_def' _ _ := zsmul_eq_mul _ _
  toRingHom := Int.castRingHom R


/-- A special case of `eq_intCast'` that happens to be true definitionally -/
@[simp]
theorem algebraMap_int_eq : algebraMap ℤ R = Int.castRingHom R :=
  rfl


instance int_algebra_subsingleton : Subsingleton (Algebra ℤ R) :=
  ⟨fun P Q => Algebra.algebra_ext P Q <| RingHom.congr_fun <| Subsingleton.elim _ _⟩


/-- If `algebraMap R A` is injective and `A` has no zero divisors,
`R`-multiples in `A` are zero only if one of the factors is zero.

Cannot be an instance because there is no `Injective (algebraMap R A)` typeclass.
-/
theorem of_algebraMap_injective [CommSemiring R] [Semiring A] [Algebra R A] [NoZeroDivisors A]
    (h : Function.Injective (algebraMap R A)) : NoZeroSMulDivisors R A :=
  ⟨fun hcx => (mul_eq_zero.mp ((smul_def _ _).symm.trans hcx)).imp_left
    (map_eq_zero_iff (algebraMap R A) h).mp⟩


theorem algebraMap_injective [CommRing R] [Ring A] [Nontrivial A] [Algebra R A]
    [NoZeroSMulDivisors R A] : Function.Injective (algebraMap R A) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Nontrivial A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ Function.Injective ⇑(algebraMap R A)
  -/
  simpa only [algebraMap_eq_smul_one'] using smul_left_injective R one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma algebraMap_eq_zero_iff [CommRing R] [Ring A] [Nontrivial A] [Algebra R A]
    [NoZeroSMulDivisors R A] {r : R} : algebraMap R A r = 0 ↔ r = 0 :=
  map_eq_zero_iff _ <| algebraMap_injective R A


@[simp]
lemma algebraMap_eq_one_iff [CommRing R] [Ring A] [Nontrivial A] [Algebra R A]
    [NoZeroSMulDivisors R A] {r : R} : algebraMap R A r = 1 ↔ r = 1 :=
  map_eq_one_iff _ <| algebraMap_injective R A


theorem _root_.NeZero.of_noZeroSMulDivisors (n : ℕ) [CommRing R] [NeZero (n : R)] [Ring A]
    [Nontrivial A] [Algebra R A] [NoZeroSMulDivisors R A] : NeZero (n : A) :=
  NeZero.nat_of_injective <| NoZeroSMulDivisors.algebraMap_injective R A


theorem iff_algebraMap_injective [CommRing R] [Ring A] [IsDomain A] [Algebra R A] :
    NoZeroSMulDivisors R A ↔ Function.Injective (algebraMap R A) :=
  ⟨@NoZeroSMulDivisors.algebraMap_injective R A _ _ _ _, NoZeroSMulDivisors.of_algebraMap_injective⟩

-- see note [lower instance priority]

instance (priority := 100) CharZero.noZeroSMulDivisors_nat [Semiring R] [NoZeroDivisors R]
    [CharZero R] : NoZeroSMulDivisors ℕ R :=
  NoZeroSMulDivisors.of_algebraMap_injective <| (algebraMap ℕ R).injective_nat

-- see note [lower instance priority]

instance (priority := 100) CharZero.noZeroSMulDivisors_int [Ring R] [NoZeroDivisors R]
    [CharZero R] : NoZeroSMulDivisors ℤ R :=
  NoZeroSMulDivisors.of_algebraMap_injective <| (algebraMap ℤ R).injective_int


theorem algebra_compatible_smul (r : R) (m : M) : r • m = (algebraMap R A) r • m := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    A : Type u_2
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    r : R
    m : M
    ⊢ Eq (HSMul.hSMul r m) (HSMul.hSMul ((algebraMap R A) r) m)
  -/
  rw [← one_smul A m, ← smul_assoc, Algebra.smul_def, mul_one, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem algebraMap_smul (r : R) (m : M) : (algebraMap R A) r • m = r • m :=
  (algebra_compatible_smul A r m).symm


theorem NoZeroSMulDivisors.trans (R A M : Type*) [CommRing R] [Ring A] [IsDomain A] [Algebra R A]
    [AddCommGroup M] [Module R M] [Module A M] [IsScalarTower R A M] [NoZeroSMulDivisors R A]
    [NoZeroSMulDivisors A M] : NoZeroSMulDivisors R M := by
  /-
    R : Type u_4
    A : Type u_5
    M : Type u_6
    inst✝⁹ : CommRing R
    inst✝⁸ : Ring A
    inst✝⁷ : IsDomain A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    inst✝¹ : NoZeroSMulDivisors R A
    inst✝ : NoZeroSMulDivisors A M
    ⊢ NoZeroSMulDivisors R M
  -/
  refine ⟨fun {r m} h => ?_⟩
  /-
    R : Type u_4
    A : Type u_5
    M : Type u_6
    inst✝⁹ : CommRing R
    inst✝⁸ : Ring A
    inst✝⁷ : IsDomain A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    inst✝¹ : NoZeroSMulDivisors R A
    inst✝ : NoZeroSMulDivisors A M
    r : R
    m : M
    h : Eq (HSMul.hSMul r m) 0
    ⊢ Or (Eq r 0) (Eq m 0)
  -/
  rw [algebra_compatible_smul A r m] at h
  /-
    R : Type u_4
    A : Type u_5
    M : Type u_6
    inst✝⁹ : CommRing R
    inst✝⁸ : Ring A
    inst✝⁷ : IsDomain A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    inst✝¹ : NoZeroSMulDivisors R A
    inst✝ : NoZeroSMulDivisors A M
    r : R
    m : M
    h : Eq (HSMul.hSMul ((algebraMap R A) r) m) 0
    ⊢ Or (Eq r 0) (Eq m 0)
  -/
  rcases smul_eq_zero.1 h with H | H
  · have : Function.Injective (algebraMap R A) :=
      NoZeroSMulDivisors.iff_algebraMap_injective.1 inferInstance
    /-
      case inl
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : IsDomain A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : Module A M
      inst✝² : IsScalarTower R A M
      inst✝¹ : NoZeroSMulDivisors R A
      inst✝ : NoZeroSMulDivisors A M
      r : R
      m : M
      h : Eq (HSMul.hSMul ((algebraMap R A) r) m) 0
      H : Eq ((algebraMap R A) r) 0
      this : Function.Injective ⇑(algebraMap R A)
      ⊢ Or (Eq r 0) (Eq m 0)
    -/
    left
    /-
      case inl.h
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : IsDomain A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : Module A M
      inst✝² : IsScalarTower R A M
      inst✝¹ : NoZeroSMulDivisors R A
      inst✝ : NoZeroSMulDivisors A M
      r : R
      m : M
      h : Eq (HSMul.hSMul ((algebraMap R A) r) m) 0
      H : Eq ((algebraMap R A) r) 0
      this : Function.Injective ⇑(algebraMap R A)
      ⊢ Eq r 0
    -/
    exact (injective_iff_map_eq_zero _).1 this _ H
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : IsDomain A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : Module A M
      inst✝² : IsScalarTower R A M
      inst✝¹ : NoZeroSMulDivisors R A
      inst✝ : NoZeroSMulDivisors A M
      r : R
      m : M
      h : Eq (HSMul.hSMul ((algebraMap R A) r) m) 0
      H : Eq m 0
      ⊢ Or (Eq r 0) (Eq m 0)
    -/
  · right
    /-
      case inr.h
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : IsDomain A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : Module A M
      inst✝² : IsScalarTower R A M
      inst✝¹ : NoZeroSMulDivisors R A
      inst✝ : NoZeroSMulDivisors A M
      r : R
      m : M
      h : Eq (HSMul.hSMul ((algebraMap R A) r) m) 0
      H : Eq m 0
      ⊢ Eq m 0
    -/
    exact H
    /-
      🎉 no goals
    -/


instance (priority := 120) IsScalarTower.to_smulCommClass : SMulCommClass R A M :=
  ⟨fun r a m => by
    rw [algebra_compatible_smul A r (a • m), smul_smul, Algebra.commutes, mul_smul, ←
      algebra_compatible_smul]⟩

-- see Note [lower instance priority]
-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980, as it is a very common path

instance (priority := 110) IsScalarTower.to_smulCommClass' : SMulCommClass A R M :=
  SMulCommClass.symm _ _ _

-- see Note [lower instance priority]

instance (priority := 200) Algebra.to_smulCommClass {R A} [CommSemiring R] [Semiring A]
    [Algebra R A] : SMulCommClass R A A :=
  IsScalarTower.to_smulCommClass


theorem smul_algebra_smul_comm (r : R) (a : A) (m : M) : a • r • m = r • a • m :=
  smul_comm _ _ _


/-- `A`-linearly coerce an `R`-linear map from `M` to `A` to a function, given an algebra `A` over
a commutative semiring `R` and `M` a module over `R`. -/
def ltoFun (R : Type u) (M : Type v) (A : Type w) [CommSemiring R] [AddCommMonoid M] [Module R M]
    [CommSemiring A] [Algebra R A] : (M →ₗ[R] A) →ₗ[A] M → A where
  toFun f := f.toFun
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem LinearMap.ker_restrictScalars (f : M →ₗ[S] N) :
    LinearMap.ker (f.restrictScalars R) = f.ker.restrictScalars R :=
  rfl


/-- If there is a linear map `f : A →ₗ[R] B` that preserves `1`, then `algebraMap R B r` is
invertible when `algebraMap R A r` is. -/
abbrev Invertible.algebraMapOfInvertibleAlgebraMap (f : A →ₗ[R] B) (hf : f 1 = 1) {r : R}
    (h : Invertible (algebraMap R A r)) : Invertible (algebraMap R B r) where
  invOf := f ⅟(algebraMap R A r)
  invOf_mul_self := by rw [← Algebra.commutes, ← Algebra.smul_def, ← map_smul, Algebra.smul_def,
    mul_invOf_self, hf]
                       /-
                         R : Type u_1
                         A : Type u_2
                         B : Type u_3
                         inst✝⁴ : CommSemiring R
                         inst✝³ : Semiring A
                         inst✝² : Semiring B
                         inst✝¹ : Algebra R A
                         inst✝ : Algebra R B
                         f : LinearMap (RingHom.id R) A B
                         hf : Eq (f 1) 1
                         r : R
                         h : Invertible ((algebraMap R A) r)
                         ⊢ Eq (HMul.hMul ((algebraMap R B) r) (f (Invertible.invOf ((algebraMap R A) r) …
                       -/
  mul_invOf_self := by rw [← Algebra.smul_def, ← map_smul, Algebra.smul_def, mul_invOf_self, hf]
                       /-
                         🎉 no goals
                       -/


/-- If there is a linear map `f : A →ₗ[R] B` that preserves `1`, then `algebraMap R B r` is
a unit when `algebraMap R A r` is. -/
lemma IsUnit.algebraMap_of_algebraMap (f : A →ₗ[R] B) (hf : f 1 = 1) {r : R}
    (h : IsUnit (algebraMap R A r)) : IsUnit (algebraMap R B r) :=
  let ⟨i⟩ := nonempty_invertible h
  letI := Invertible.algebraMapOfInvertibleAlgebraMap f hf i
  isUnit_of_invertible _


/-- If `E` is an `F`-algebra, and there exists an injective `F`-linear map from `F` to `E`,
then the algebra map from `F` to `E` is also injective. -/
theorem injective_algebraMap_of_linearMap (hb : Function.Injective b) :
    Function.Injective (algebraMap F E) := fun x y e ↦ hb <| by
  rw [← mul_one x, ← mul_one y, ← smul_eq_mul, ← smul_eq_mul,
    map_smul, map_smul, Algebra.smul_def, Algebra.smul_def, e]


/-- If `E` is an `F`-algebra, and there exists a surjective `F`-linear map from `F` to `E`,
then the algebra map from `F` to `E` is also surjective. -/
theorem surjective_algebraMap_of_linearMap (hb : Function.Surjective b) :
    Function.Surjective (algebraMap F E) := fun x ↦ by
  /-
    F : Type u_1
    E : Type u_2
    inst✝² : CommSemiring F
    inst✝¹ : Semiring E
    inst✝ : Algebra F E
    b : LinearMap (RingHom.id F) F E
    hb : Function.Surjective ⇑b
    x : E
    ⊢ Exists fun a => Eq ((algebraMap F E) a) x
  -/
  obtain ⟨x, rfl⟩ := hb x
  /-
    case intro
    F : Type u_1
    E : Type u_2
    inst✝² : CommSemiring F
    inst✝¹ : Semiring E
    inst✝ : Algebra F E
    b : LinearMap (RingHom.id F) F E
    hb : Function.Surjective ⇑b
    x : F
    ⊢ Exists fun a => Eq ((algebraMap F E) a) (b x)
  -/
  obtain ⟨y, hy⟩ := hb (b 1 * b 1)
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    inst✝² : CommSemiring F
    inst✝¹ : Semiring E
    inst✝ : Algebra F E
    b : LinearMap (RingHom.id F) F E
    hb : Function.Surjective ⇑b
    x y : F
    hy : Eq (b y) (HMul.hMul (b 1) (b 1))
    ⊢ Exists fun a => Eq ((algebraMap F E) a) (b x)
  -/
  refine ⟨x * y, ?_⟩
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    inst✝² : CommSemiring F
    inst✝¹ : Semiring E
    inst✝ : Algebra F E
    b : LinearMap (RingHom.id F) F E
    hb : Function.Surjective ⇑b
    x y : F
    hy : Eq (b y) (HMul.hMul (b 1) (b 1))
    ⊢ Eq ((algebraMap F E) (HMul.hMul x y)) (b x)
  -/
  obtain ⟨z, hz⟩ := hb 1
  /-
    case intro.intro.intro
    F : Type u_1
    E : Type u_2
    inst✝² : CommSemiring F
    inst✝¹ : Semiring E
    inst✝ : Algebra F E
    b : LinearMap (RingHom.id F) F E
    hb : Function.Surjective ⇑b
    x y : F
    hy : Eq (b y) (HMul.hMul (b 1) (b 1))
    z : F
    hz : Eq (b z) 1
    ⊢ Eq ((algebraMap F E) (HMul.hMul x y)) (b x)
  -/
  apply_fun (x • z • ·) at hy
  rwa [← map_smul, smul_eq_mul, mul_comm, ← smul_mul_assoc, ← map_smul _ z, smul_eq_mul, mul_one,
    ← smul_eq_mul, map_smul, hz, one_mul, ← map_smul, smul_eq_mul, mul_one, smul_smul,
    ← Algebra.algebraMap_eq_smul_one] at hy


/-- If `E` is an `F`-algebra, and there exists a bijective `F`-linear map from `F` to `E`,
then the algebra map from `F` to `E` is also bijective.

NOTE: The same result can also be obtained if there are two `F`-linear maps from `F` to `E`,
one is injective, the other one is surjective. In this case, use
`injective_algebraMap_of_linearMap` and `surjective_algebraMap_of_linearMap` separately. -/
theorem bijective_algebraMap_of_linearMap (hb : Function.Bijective b) :
    Function.Bijective (algebraMap F E) :=
  ⟨injective_algebraMap_of_linearMap b hb.1, surjective_algebraMap_of_linearMap b hb.2⟩


/-- If `E` is an `F`-algebra, there exists an `F`-linear isomorphism from `F` to `E` (namely,
`E` is a free `F`-module of rank one), then the algebra map from `F` to `E` is bijective. -/
theorem bijective_algebraMap_of_linearEquiv (b : F ≃ₗ[F] E) :
    Function.Bijective (algebraMap F E) :=
  bijective_algebraMap_of_linearMap _ b.bijective


/-- If `R →+* S` is surjective, then `S`-linear maps between modules are exactly `R`-linear maps. -/
def LinearMap.extendScalarsOfSurjectiveEquiv (h : Function.Surjective (algebraMap R S)) :
    (M →ₗ[R] N) ≃ₗ[R] (M →ₗ[S] N) where
                                                  /-
                                                    R : Type ?u.96050
                                                    S : Type ?u.96053
                                                    inst✝¹⁰ : CommSemiring R
                                                    inst✝⁹ : Semiring S
                                                    inst✝⁸ : Algebra R S
                                                    M : Type ?u.96078
                                                    N : Type ?u.96081
                                                    inst✝⁷ : AddCommMonoid M
                                                    inst✝⁶ : AddCommMonoid N
                                                    inst✝⁵ : Module R M
                                                    inst✝⁴ : Module S M
                                                    inst✝³ : IsScalarTower R S M
                                                    inst✝² : Module R N
                                                    inst✝¹ : Module S N
                                                    inst✝ : IsScalarTower R S N
                                                    h : Function.Surjective ⇑(algebraMap R S)
                                                    f : LinearMap (RingHom.id R) M N
                                                    r : S
                                                    x : M
                                                    ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id S) r) (__ …
                                                  -/
  toFun f := { __ := f, map_smul' := fun r x ↦ by obtain ⟨r, rfl⟩ := h r; simp }
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun f := f.restrictScalars S
  left_inv _ := rfl
  right_inv _ := rfl


/-- If `R →+* S` is surjective, then `R`-linear maps are also `S`-linear. -/
abbrev LinearMap.extendScalarsOfSurjective (h : Function.Surjective (algebraMap R S))
    (l : M →ₗ[R] N) : M →ₗ[S] N :=
  extendScalarsOfSurjectiveEquiv h l


/-- If `R →+* S` is surjective, then `R`-linear isomorphisms are also `S`-linear. -/
def LinearEquiv.extendScalarsOfSurjective (h : Function.Surjective (algebraMap R S))
    (f : M ≃ₗ[R] N) : M ≃ₗ[S] N where
  __ := f
                      /-
                        R : Type ?u.104505
                        S : Type ?u.104508
                        inst✝¹⁰ : CommSemiring R
                        inst✝⁹ : Semiring S
                        inst✝⁸ : Algebra R S
                        M : Type ?u.104533
                        N : Type ?u.104536
                        inst✝⁷ : AddCommMonoid M
                        inst✝⁶ : AddCommMonoid N
                        inst✝⁵ : Module R M
                        inst✝⁴ : Module S M
                        inst✝³ : IsScalarTower R S M
                        inst✝² : Module R N
                        inst✝¹ : Module S N
                        inst✝ : IsScalarTower R S N
                        h : Function.Surjective ⇑(algebraMap R S)
                        f : LinearEquiv (RingHom.id R) M N
                        r : S
                        x : M
                        ⊢ Eq ((↑__spread✝⁻⁰).toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id S) r)  …
                      -/
  map_smul' r x := by obtain ⟨r, rfl⟩ := h r; simp
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma LinearMap.extendScalarsOfSurjective_apply (l : M →ₗ[R] N) (x) :
    l.extendScalarsOfSurjective h x = l x := rfl


@[simp]
lemma LinearEquiv.extendScalarsOfSurjective_apply (f : M ≃ₗ[R] N) (x) :
    f.extendScalarsOfSurjective h x = f x := rfl


@[simp]
lemma LinearEquiv.extendScalarsOfSurjective_symm (f : M ≃ₗ[R] N) :
    (f.extendScalarsOfSurjective h).symm = f.symm.extendScalarsOfSurjective h := rfl


@[norm_cast]
theorem coe_prod (a : ι → R) : (↑(∏ i ∈ s, a i : R) : A) = ∏ i ∈ s, (↑(a i) : A) :=
  map_prod (algebraMap R A) a s


@[norm_cast]
theorem coe_sum (a : ι → R) : ↑(∑ i ∈ s, a i) = ∑ i ∈ s, (↑(a i) : A) :=
  map_sum (algebraMap R A) a s


