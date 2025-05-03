/-- `AddGroupConeClass S G` says that `S` is a type of cones in `G`. -/
class AddGroupConeClass (S : Type*) (G : outParam Type*) [AddCommGroup G] [SetLike S G]
    extends AddSubmonoidClass S G : Prop where
  eq_zero_of_mem_of_neg_mem {C : S} {a : G} : a ∈ C → -a ∈ C → a = 0


/-- `GroupConeClass S G` says that `S` is a type of cones in `G`. -/
@[to_additive]
class GroupConeClass (S : Type*) (G : outParam Type*) [CommGroup G] [SetLike S G] extends
    SubmonoidClass S G : Prop where
  eq_one_of_mem_of_inv_mem {C : S} {a : G} : a ∈ C → a⁻¹ ∈ C → a = 1


/-- A (positive) cone in an abelian group is a submonoid that
does not contain both `a` and `-a` for any nonzero `a`.
This is equivalent to being the set of non-negative elements of
some order making the group into a partially ordered group. -/
structure AddGroupCone (G : Type*) [AddCommGroup G] extends AddSubmonoid G where
  eq_zero_of_mem_of_neg_mem' {a} : a ∈ carrier → -a ∈ carrier → a = 0


/-- A (positive) cone in an abelian group is a submonoid that
does not contain both `a` and `a⁻¹` for any non-identity `a`.
This is equivalent to being the set of elements that are at least 1 in
some order making the group into a partially ordered group. -/
@[to_additive]
structure GroupCone (G : Type*) [CommGroup G] extends Submonoid G where
  eq_one_of_mem_of_inv_mem' {a} : a ∈ carrier → a⁻¹ ∈ carrier → a = 1


@[to_additive]
instance GroupCone.instSetLike (G : Type*) [CommGroup G] : SetLike (GroupCone G) G where
  coe C := C.carrier
                             /-
                               G : Type u_1
                               inst✝ : CommGroup G
                               p q : GroupCone G
                               h : Eq ((fun C => C.carrier) p) ((fun C => C.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.ext' h
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
instance GroupCone.instGroupConeClass (G : Type*) [CommGroup G] :
    GroupConeClass (GroupCone G) G where
  mul_mem {C} := C.mul_mem'
  one_mem {C} := C.one_mem'
  eq_one_of_mem_of_inv_mem {C} := C.eq_one_of_mem_of_inv_mem'


/-- Typeclass for maximal additive cones. -/
class IsMaxCone {S G : Type*} [AddCommGroup G] [SetLike S G] (C : S) : Prop where
  mem_or_neg_mem (a : G) : a ∈ C ∨ -a ∈ C


/-- Typeclass for maximal multiplicative cones. -/
@[to_additive IsMaxCone]
class IsMaxMulCone {S G : Type*} [CommGroup G] [SetLike S G] (C : S) : Prop where
  mem_or_inv_mem (a : G) : a ∈ C ∨ a⁻¹ ∈ C


variable (H) in
/-- The cone of elements that are at least 1. -/
@[to_additive "The cone of non-negative elements."]
def oneLE : GroupCone H where
  __ := Submonoid.oneLE H
                                      /-
                                        H : Type u_1
                                        inst✝ : OrderedCommGroup H
                                        a✝ a : H
                                        ⊢ Membership.mem __spread✝⁻⁰.carrier a → Membership.mem __spread✝⁻⁰.carrier (I …
                                      -/
  eq_one_of_mem_of_inv_mem' {a} := by simpa using ge_antisymm
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive (attr := simp)]
lemma oneLE_toSubmonoid : (oneLE H).toSubmonoid = .oneLE H := rfl

@[to_additive (attr := simp)]
lemma mem_oneLE : a ∈ oneLE H ↔ 1 ≤ a := Iff.rfl

@[to_additive (attr := simp, norm_cast)]
lemma coe_oneLE : oneLE H = {x : H | 1 ≤ x} := rfl


@[to_additive nonneg.isMaxCone]
instance oneLE.isMaxMulCone {H : Type*} [LinearOrderedCommGroup H] : IsMaxMulCone (oneLE H) where
                       /-
                         H✝ : Type u_1
                         inst✝¹ : OrderedCommGroup H✝
                         a : H✝
                         H : Type u_2
                         inst✝ : LinearOrderedCommGroup H
                         ⊢ ∀ (a : H), Or (Membership.mem (GroupCone.oneLE H) a) (Membership.mem (GroupC …
                       -/
  mem_or_inv_mem := by simpa using le_total 1
                       /-
                         🎉 no goals
                       -/


/-- Construct a partially ordered abelian group by designating a cone in an abelian group. -/
@[to_additive (attr := reducible)
"Construct a partially ordered abelian group by designating a cone in an abelian group."]
def OrderedCommGroup.mkOfCone [GroupConeClass S G] :
    OrderedCommGroup G where
  le a b := b / a ∈ C
                  /-
                    S : Type u_1
                    G : Type u_2
                    inst✝² : CommGroup G
                    inst✝¹ : SetLike S G
                    C : S
                    inst✝ : GroupConeClass S G
                    a : G
                    ⊢ LE.le a a
                  -/
  le_refl a := by simp [one_mem]
                  /-
                    🎉 no goals
                  -/
                               /-
                                 S : Type u_1
                                 G : Type u_2
                                 inst✝² : CommGroup G
                                 inst✝¹ : SetLike S G
                                 C : S
                                 inst✝ : GroupConeClass S G
                                 a b c : G
                                 nab : LE.le a b
                                 nbc : LE.le b c
                                 ⊢ LE.le a c
                               -/
  le_trans a b c nab nbc := by simpa using mul_mem nbc nab
                               /-
                                 🎉 no goals
                               -/
  le_antisymm a b nab nba := by
    /-
      S : Type u_1
      G : Type u_2
      inst✝² : CommGroup G
      inst✝¹ : SetLike S G
      C : S
      inst✝ : GroupConeClass S G
      a b : G
      nab : LE.le a b
      nba : LE.le b a
      ⊢ Eq a b
    -/
    simpa [div_eq_one, eq_comm] using eq_one_of_mem_of_inv_mem nab (by simpa using nba)
    /-
      🎉 no goals
    -/
                                  /-
                                    S : Type u_1
                                    G : Type u_2
                                    inst✝² : CommGroup G
                                    inst✝¹ : SetLike S G
                                    C : S
                                    inst✝ : GroupConeClass S G
                                    a b : G
                                    nab : LE.le a b
                                    c : G
                                    ⊢ LE.le (HMul.hMul c a) (HMul.hMul c b)
                                  -/
  mul_le_mul_left a b nab c := by simpa using nab
                                  /-
                                    🎉 no goals
                                  -/


/-- Construct a linearly ordered abelian group by designating a maximal cone in an abelian group. -/
@[to_additive (attr := reducible)
"Construct a linearly ordered abelian group by designating a maximal cone in an abelian group."]
def LinearOrderedCommGroup.mkOfCone
    [GroupConeClass S G] [IsMaxMulCone C] (dec : DecidablePred (· ∈ C)) :
    LinearOrderedCommGroup G where
  __ := OrderedCommGroup.mkOfCone C
                     /-
                       S : Type u_1
                       G : Type u_2
                       inst✝³ : CommGroup G
                       inst✝² : SetLike S G
                       C : S
                       inst✝¹ : GroupConeClass S G
                       inst✝ : IsMaxMulCone C
                       dec : DecidablePred fun x => Membership.mem C x
                       a b : G
                       ⊢ Or (LE.le a b) (LE.le b a)
                     -/
  le_total a b := by simpa using mem_or_inv_mem (b / a)
                     /-
                       🎉 no goals
                     -/
  decidableLE _ _ := dec _

