/-- `NonUnitalSubsemiringClass S R` states that `S` is a type of subsets `s ⊆ R` that
are both an additive submonoid and also a multiplicative subsemigroup. -/
class NonUnitalSubsemiringClass (S : Type*) (R : outParam (Type u)) [NonUnitalNonAssocSemiring R]
  [SetLike S R] extends AddSubmonoidClass S R : Prop where
  mul_mem : ∀ {s : S} {a b : R}, a ∈ s → b ∈ s → a * b ∈ s

-- See note [lower instance priority]

instance (priority := 100) NonUnitalSubsemiringClass.mulMemClass (S : Type*) (R : Type u)
    [NonUnitalNonAssocSemiring R] [SetLike S R] [h : NonUnitalSubsemiringClass S R] :
    MulMemClass S R :=
  { h with }


/-- A non-unital subsemiring of a `NonUnitalNonAssocSemiring` inherits a
`NonUnitalNonAssocSemiring` structure -/
instance (priority := 75) toNonUnitalNonAssocSemiring : NonUnitalNonAssocSemiring s :=
                                                                      /-
                                                                        R : Type u
                                                                        S : Type v
                                                                        T : Type w
                                                                        inst✝² : NonUnitalNonAssocSemiring R
                                                                        inst✝¹ : SetLike S R
                                                                        inst✝ : NonUnitalSubsemiringClass S R
                                                                        s : S
                                                                        ⊢ ∀ (x y : Subtype fun x => Membership.mem s x), Eq (↑(HAdd.hAdd x y)) (HAdd.h …
                                                                      -/
  Subtype.coe_injective.nonUnitalNonAssocSemiring Subtype.val rfl (by simp) (fun _ _ => rfl)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    fun _ _ => rfl


instance noZeroDivisors [NoZeroDivisors R] : NoZeroDivisors s :=
  Subtype.coe_injective.noZeroDivisors Subtype.val rfl fun _ _ => rfl


/-- The natural non-unital ring hom from a non-unital subsemiring of a non-unital semiring `R` to
`R`. -/
def subtype : s →ₙ+* R :=
  { AddSubmonoidClass.subtype s, MulMemClass.subtype s with toFun := (↑) }


@[simp]
theorem coeSubtype : (subtype s : s → R) = ((↑) : s → R) :=
  rfl


/-- A non-unital subsemiring of a `NonUnitalSemiring` is a `NonUnitalSemiring`. -/
instance toNonUnitalSemiring {R} [NonUnitalSemiring R] [SetLike S R]
    [NonUnitalSubsemiringClass S R] : NonUnitalSemiring s :=
                                                              /-
                                                                R✝ : Type u
                                                                S : Type v
                                                                T : Type w
                                                                inst✝⁵ : NonUnitalNonAssocSemiring R✝
                                                                inst✝⁴ : SetLike S R✝
                                                                inst✝³ : NonUnitalSubsemiringClass S R✝
                                                                s : S
                                                                R : Type ?u.6444
                                                                inst✝² : NonUnitalSemiring R
                                                                inst✝¹ : SetLike S R
                                                                inst✝ : NonUnitalSubsemiringClass S R
                                                                ⊢ ∀ (x y : Subtype fun x => Membership.mem s x), Eq (↑(HAdd.hAdd x y)) (HAdd.h …
                                                              -/
  Subtype.coe_injective.nonUnitalSemiring Subtype.val rfl (by simp) (fun _ _ => rfl) fun _ _ => rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- A non-unital subsemiring of a `NonUnitalCommSemiring` is a `NonUnitalCommSemiring`. -/
instance toNonUnitalCommSemiring {R} [NonUnitalCommSemiring R] [SetLike S R]
    [NonUnitalSubsemiringClass S R] : NonUnitalCommSemiring s :=
                                                                  /-
                                                                    R✝ : Type u
                                                                    S : Type v
                                                                    T : Type w
                                                                    inst✝⁵ : NonUnitalNonAssocSemiring R✝
                                                                    inst✝⁴ : SetLike S R✝
                                                                    inst✝³ : NonUnitalSubsemiringClass S R✝
                                                                    s : S
                                                                    R : Type ?u.8642
                                                                    inst✝² : NonUnitalCommSemiring R
                                                                    inst✝¹ : SetLike S R
                                                                    inst✝ : NonUnitalSubsemiringClass S R
                                                                    ⊢ ∀ (x y : Subtype fun x => Membership.mem s x), Eq (↑(HAdd.hAdd x y)) (HAdd.h …
                                                                  -/
  Subtype.coe_injective.nonUnitalCommSemiring Subtype.val rfl (by simp) (fun _ _ => rfl)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    fun _ _ => rfl


/-- A non-unital subsemiring of a non-unital semiring `R` is a subset `s` that is both an additive
submonoid and a semigroup. -/
structure NonUnitalSubsemiring (R : Type u) [NonUnitalNonAssocSemiring R] extends AddSubmonoid R,
  Subsemigroup R


instance : SetLike (NonUnitalSubsemiring R) R where
  coe s := s.carrier
                             /-
                               R : Type u
                               S : Type v
                               T : Type w
                               inst✝ : NonUnitalNonAssocSemiring R
                               p q : NonUnitalSubsemiring R
                               h : Eq ((fun s => s.carrier) p) ((fun s => s.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.coe_injective' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance : NonUnitalSubsemiringClass (NonUnitalSubsemiring R) R where
  zero_mem {s} := AddSubmonoid.zero_mem' s.toAddSubmonoid
  add_mem {s} := AddSubsemigroup.add_mem' s.toAddSubmonoid.toAddSubsemigroup
  mul_mem {s} := mul_mem' s


theorem mem_carrier {s : NonUnitalSubsemiring R} {x : R} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


/-- Two non-unital subsemirings are equal if they have the same elements. -/
@[ext]
theorem ext {S T : NonUnitalSubsemiring R} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


/-- Copy of a non-unital subsemiring with a new `carrier` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (S : NonUnitalSubsemiring R) (s : Set R) (hs : s = ↑S) :
    NonUnitalSubsemiring R :=
  { S.toAddSubmonoid.copy s hs, S.toSubsemigroup.copy s hs with carrier := s }


@[simp]
theorem coe_copy (S : NonUnitalSubsemiring R) (s : Set R) (hs : s = ↑S) :
    (S.copy s hs : Set R) = s :=
  rfl


theorem copy_eq (S : NonUnitalSubsemiring R) (s : Set R) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


theorem toSubsemigroup_injective :
    Function.Injective (toSubsemigroup : NonUnitalSubsemiring R → Subsemigroup R)
  | _, _, h => ext (SetLike.ext_iff.mp h : _)


theorem toAddSubmonoid_injective :
    Function.Injective (toAddSubmonoid : NonUnitalSubsemiring R → AddSubmonoid R)
  | _, _, h => ext (SetLike.ext_iff.mp h : _)


/-- Construct a `NonUnitalSubsemiring R` from a set `s`, a subsemigroup `sg`, and an additive
submonoid `sa` such that `x ∈ s ↔ x ∈ sg ↔ x ∈ sa`. -/
protected def mk' (s : Set R) (sg : Subsemigroup R) (hg : ↑sg = s) (sa : AddSubmonoid R)
    (ha : ↑sa = s) : NonUnitalSubsemiring R where
  carrier := s
                  /-
                    R : Type u
                    S : Type v
                    T : Type w
                    inst✝ : NonUnitalNonAssocSemiring R
                    s : Set R
                    sg : Subsemigroup R
                    hg : Eq (↑sg) s
                    sa : AddSubmonoid R
                    ha : Eq (↑sa) s
                    ⊢ Membership.mem { carrier := s, add_mem' := ⋯ }.carrier 0
                  -/
                 /-
                   R : Type u
                   S : Type v
                   T : Type w
                   inst✝ : NonUnitalNonAssocSemiring R
                   s : Set R
                   sg : Subsemigroup R
                   hg : Eq (↑sg) s
                   sa : AddSubmonoid R
                   ha : Eq (↑sa) s
                   ⊢ ∀ {a b : R}, Membership.mem s a → Membership.mem s b → Membership.mem s (HAd …
                 -/
  zero_mem' := by subst ha; exact sa.zero_mem
                           /-
                             🎉 no goals
                           -/
                            /-
                              🎉 no goals
                            -/
  add_mem' := by subst ha; exact sa.add_mem
                 /-
                   R : Type u
                   S : Type v
                   T : Type w
                   inst✝ : NonUnitalNonAssocSemiring R
                   s : Set R
                   sg : Subsemigroup R
                   hg : Eq (↑sg) s
                   sa : AddSubmonoid R
                   ha : Eq (↑sa) s
                   ⊢ ∀ {a b : R}, Membership.mem { carrier := s, add_mem' := ⋯, zero_mem' := ⋯ }. …
                 -/
  mul_mem' := by subst hg; exact sg.mul_mem
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem coe_mk' {s : Set R} {sg : Subsemigroup R} (hg : ↑sg = s) {sa : AddSubmonoid R}
    (ha : ↑sa = s) : (NonUnitalSubsemiring.mk' s sg hg sa ha : Set R) = s :=
  rfl


@[simp]
theorem mem_mk' {s : Set R} {sg : Subsemigroup R} (hg : ↑sg = s) {sa : AddSubmonoid R}
    (ha : ↑sa = s) {x : R} : x ∈ NonUnitalSubsemiring.mk' s sg hg sa ha ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem mk'_toSubsemigroup {s : Set R} {sg : Subsemigroup R} (hg : ↑sg = s) {sa : AddSubmonoid R}
    (ha : ↑sa = s) : (NonUnitalSubsemiring.mk' s sg hg sa ha).toSubsemigroup = sg :=
  SetLike.coe_injective hg.symm


@[simp]
theorem mk'_toAddSubmonoid {s : Set R} {sg : Subsemigroup R} (hg : ↑sg = s) {sa : AddSubmonoid R}
    (ha : ↑sa = s) : (NonUnitalSubsemiring.mk' s sg hg sa ha).toAddSubmonoid = sa :=
  SetLike.coe_injective ha.symm


@[simp, norm_cast]
theorem coe_zero : ((0 : s) : R) = (0 : R) :=
  rfl


@[simp, norm_cast]
theorem coe_add (x y : s) : ((x + y : s) : R) = (x + y : R) :=
  rfl


@[simp, norm_cast]
theorem coe_mul (x y : s) : ((x * y : s) : R) = (x * y : R) :=
  rfl


@[simp high]
theorem mem_toSubsemigroup {s : NonUnitalSubsemiring R} {x : R} : x ∈ s.toSubsemigroup ↔ x ∈ s :=
  Iff.rfl


@[simp high]
theorem coe_toSubsemigroup (s : NonUnitalSubsemiring R) : (s.toSubsemigroup : Set R) = s :=
  rfl


@[simp]
theorem mem_toAddSubmonoid {s : NonUnitalSubsemiring R} {x : R} : x ∈ s.toAddSubmonoid ↔ x ∈ s :=
  Iff.rfl


@[simp]
theorem coe_toAddSubmonoid (s : NonUnitalSubsemiring R) : (s.toAddSubmonoid : Set R) = s :=
  rfl


/-- The non-unital subsemiring `R` of the non-unital semiring `R`. -/
instance : Top (NonUnitalSubsemiring R) :=
  ⟨{ (⊤ : Subsemigroup R), (⊤ : AddSubmonoid R) with }⟩


@[simp]
theorem mem_top (x : R) : x ∈ (⊤ : NonUnitalSubsemiring R) :=
  Set.mem_univ x


@[simp]
theorem coe_top : ((⊤ : NonUnitalSubsemiring R) : Set R) = Set.univ :=
  rfl


instance : Bot (NonUnitalSubsemiring R) :=
  ⟨{  carrier := {0}
                                /-
                                  R : Type u
                                  S : Type v
                                  T : Type w
                                  inst✝ : NonUnitalNonAssocSemiring R
                                  a✝ b✝ : R
                                  x✝¹ : Membership.mem (Singleton.singleton 0) a✝
                                  x✝ : Membership.mem (Singleton.singleton 0) b✝
                                  ⊢ Membership.mem (Singleton.singleton 0) (HAdd.hAdd a✝ b✝)
                                -/
      add_mem' := fun _ _ => by simp_all
                                /-
                                  🎉 no goals
                                -/
      zero_mem' := Set.mem_singleton 0
                                /-
                                  R : Type u
                                  S : Type v
                                  T : Type w
                                  inst✝ : NonUnitalNonAssocSemiring R
                                  a✝ b✝ : R
                                  x✝¹ : Membership.mem { carrier := Singleton.singleton 0, add_mem' := ⋯, zero_m …
                                  x✝ : Membership.mem { carrier := Singleton.singleton 0, add_mem' := ⋯, zero_me …
                                  ⊢ Membership.mem { carrier := Singleton.singleton 0, add_mem' := ⋯, zero_mem'  …
                                -/
      mul_mem' := fun _ _ => by simp_all }⟩
                                /-
                                  🎉 no goals
                                -/


instance : Inhabited (NonUnitalSubsemiring R) :=
  ⟨⊥⟩


theorem coe_bot : ((⊥ : NonUnitalSubsemiring R) : Set R) = {0} :=
  rfl


theorem mem_bot {x : R} : x ∈ (⊥ : NonUnitalSubsemiring R) ↔ x = 0 :=
  Set.mem_singleton_iff


/-- The inf of two non-unital subsemirings is their intersection. -/
instance : Min (NonUnitalSubsemiring R) :=
  ⟨fun s t =>
    { s.toSubsemigroup ⊓ t.toSubsemigroup, s.toAddSubmonoid ⊓ t.toAddSubmonoid with
      carrier := s ∩ t }⟩


@[simp]
theorem coe_inf (p p' : NonUnitalSubsemiring R) :
    ((p ⊓ p' : NonUnitalSubsemiring R) : Set R) = (p : Set R) ∩ p' :=
  rfl


@[simp]
theorem mem_inf {p p' : NonUnitalSubsemiring R} {x : R} : x ∈ p ⊓ p' ↔ x ∈ p ∧ x ∈ p' :=
  Iff.rfl


/-- Restriction of a non-unital ring homomorphism to a non-unital subsemiring of the codomain. -/
def codRestrict (f : F) (s : S') (h : ∀ x, f x ∈ s) : R →ₙ+* s where
  toFun n := ⟨f n, h n⟩
  map_mul' x y := Subtype.eq (map_mul f x y)
  map_add' x y := Subtype.eq (map_add f x y)
  map_zero' := Subtype.eq (map_zero f)


/-- The non-unital subsemiring of elements `x : R` such that `f x = g x` -/
def eqSlocus (f g : F) : NonUnitalSubsemiring R :=
  { (f : R →ₙ* S).eqLocus (g : R →ₙ* S), (f : R →+ S).eqLocusM g with
    carrier := { x | f x = g x } }


/-- The non-unital ring homomorphism associated to an inclusion of
non-unital subsemirings. -/
def inclusion {S T : NonUnitalSubsemiring R} (h : S ≤ T) : S →ₙ+* T :=
  codRestrict (subtype S) _ fun x => h x.2


