/-- `RingConeClass S R` says that `S` is a type of cones in `R`. -/
class RingConeClass (S : Type*) (R : outParam Type*) [Ring R] [SetLike S R]
    extends AddGroupConeClass S R, SubsemiringClass S R : Prop


/-- A (positive) cone in a ring is a subsemiring that
does not contain both `a` and `-a` for any nonzero `a`.
This is equivalent to being the set of non-negative elements of
some order making the ring into a partially ordered ring. -/
structure RingCone (R : Type*) [Ring R] extends Subsemiring R, AddGroupCone R


instance RingCone.instSetLike (R : Type*) [Ring R] : SetLike (RingCone R) R where
  coe C := C.carrier
                             /-
                               R : Type u_1
                               inst✝ : Ring R
                               p q : RingCone R
                               h : Eq ((fun C => C.carrier) p) ((fun C => C.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.ext' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance RingCone.instRingConeClass (R : Type*) [Ring R] :
    RingConeClass (RingCone R) R where
  add_mem {C} := C.add_mem'
  zero_mem {C} := C.zero_mem'
  mul_mem {C} := C.mul_mem'
  one_mem {C} := C.one_mem'
  eq_zero_of_mem_of_neg_mem {C} := C.eq_zero_of_mem_of_neg_mem'


variable (T) in
/-- Construct a cone from the set of non-negative elements of a partially ordered ring. -/
def nonneg : RingCone T where
  __ := Subsemiring.nonneg T
                                       /-
                                         T : Type u_1
                                         inst✝ : OrderedRing T
                                         a✝ a : T
                                         ⊢ Membership.mem __spread✝⁻⁰.carrier a → Membership.mem __spread✝⁻⁰.carrier (N …
                                       -/
  eq_zero_of_mem_of_neg_mem' {a} := by simpa using ge_antisymm
                                       /-
                                         🎉 no goals
                                       -/


@[simp] lemma nonneg_toSubsemiring : (nonneg T).toSubsemiring = .nonneg T := rfl

@[simp] lemma nonneg_toAddGroupCone : (nonneg T).toAddGroupCone = .nonneg T := rfl

@[simp] lemma mem_nonneg : a ∈ nonneg T ↔ 0 ≤ a := Iff.rfl

@[simp, norm_cast] lemma coe_nonneg : nonneg T = {x : T | 0 ≤ x} := rfl


instance nonneg.isMaxCone {T : Type*} [LinearOrderedRing T] : IsMaxCone (nonneg T) where
  mem_or_neg_mem := mem_or_neg_mem (C := AddGroupCone.nonneg T)


/-- Construct a partially ordered ring by designating a cone in a ring.
Warning: using this def as a constructor in an instance can lead to diamonds
due to non-customisable field: `lt`. -/
@[reducible] def OrderedRing.mkOfCone [RingConeClass S R] : OrderedRing R where
  __ := ‹Ring R›
  __ := OrderedAddCommGroup.mkOfCone C
                               /-
                                 S : Type u_1
                                 R : Type u_2
                                 inst✝² : Ring R
                                 inst✝¹ : SetLike S R
                                 C : S
                                 inst✝ : RingConeClass S R
                                 ⊢ Membership.mem C (HSub.hSub 1 0)
                               -/
  zero_le_one := show _ ∈ C by simpa using one_mem C
                               /-
                                 🎉 no goals
                               -/
                                          /-
                                            S : Type u_1
                                            R : Type u_2
                                            inst✝² : Ring R
                                            inst✝¹ : SetLike S R
                                            C : S
                                            inst✝ : RingConeClass S R
                                            x y : R
                                            xnn : LE.le 0 x
                                            ynn : LE.le 0 y
                                            ⊢ Membership.mem C (HSub.hSub (HMul.hMul x y) 0)
                                          -/
  mul_nonneg x y xnn ynn := show _ ∈ C by simpa using mul_mem xnn ynn
                                          /-
                                            🎉 no goals
                                          -/


/-- Construct a linearly ordered domain by designating a maximal cone in a domain.
Warning: using this def as a constructor in an instance can lead to diamonds
due to non-customisable fields: `lt`, `decidableLT`, `decidableEq`, `compare`. -/
@[reducible] def LinearOrderedRing.mkOfCone
    [IsDomain R] [RingConeClass S R] [IsMaxCone C]
    (dec : DecidablePred (· ∈ C)) : LinearOrderedRing R where
  __ := OrderedRing.mkOfCone C
  __ := OrderedRing.toStrictOrderedRing R
                     /-
                       S : Type u_1
                       R : Type u_2
                       inst✝⁴ : Ring R
                       inst✝³ : SetLike S R
                       C : S
                       inst✝² : IsDomain R
                       inst✝¹ : RingConeClass S R
                       inst✝ : IsMaxCone C
                       dec : DecidablePred fun x => Membership.mem C x
                       a b : R
                       ⊢ Or (LE.le a b) (LE.le b a)
                     -/
  le_total a b := by simpa using mem_or_neg_mem (b - a)
                     /-
                       🎉 no goals
                     -/
  decidableLE _ _ := dec _

