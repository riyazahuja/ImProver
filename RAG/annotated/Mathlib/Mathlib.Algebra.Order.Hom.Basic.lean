/-- `NonnegHomClass F α β` states that `F` is a type of nonnegative morphisms. -/
class NonnegHomClass (F : Type*) (α β : outParam Type*) [Zero β] [LE β] [FunLike F α β] : Prop where
  /-- the image of any element is non negative. -/
  apply_nonneg (f : F) : ∀ a, 0 ≤ f a


/-- `SubadditiveHomClass F α β` states that `F` is a type of subadditive morphisms. -/
class SubadditiveHomClass (F : Type*) (α β : outParam Type*)
    [Add α] [Add β] [LE β] [FunLike F α β] : Prop where
  /-- the image of a sum is less or equal than the sum of the images. -/
  map_add_le_add (f : F) : ∀ a b, f (a + b) ≤ f a + f b


/-- `SubmultiplicativeHomClass F α β` states that `F` is a type of submultiplicative morphisms. -/
@[to_additive SubadditiveHomClass]
class SubmultiplicativeHomClass (F : Type*) (α β : outParam (Type*)) [Mul α] [Mul β] [LE β]
    [FunLike F α β] : Prop where
  /-- the image of a product is less or equal than the product of the images. -/
  map_mul_le_mul (f : F) : ∀ a b, f (a * b) ≤ f a * f b


/-- `MulLEAddHomClass F α β` states that `F` is a type of subadditive morphisms. -/
@[to_additive SubadditiveHomClass]
class MulLEAddHomClass (F : Type*) (α β : outParam Type*) [Mul α] [Add β] [LE β] [FunLike F α β] :
    Prop where
  /-- the image of a product is less or equal than the sum of the images. -/
  map_mul_le_add (f : F) : ∀ a b, f (a * b) ≤ f a + f b


/-- `NonarchimedeanHomClass F α β` states that `F` is a type of non-archimedean morphisms. -/
class NonarchimedeanHomClass (F : Type*) (α β : outParam Type*)
    [Add α] [LinearOrder β] [FunLike F α β] : Prop where
  /-- the image of a sum is less or equal than the maximum of the images. -/
  map_add_le_max (f : F) : ∀ a b, f (a + b) ≤ max (f a) (f b)


attribute [simp] apply_nonneg


@[to_additive]
theorem le_map_mul_map_div [Group α] [CommSemigroup β] [LE β] [SubmultiplicativeHomClass F α β]
    (f : F) (a b : α) : f a ≤ f b * f (a / b) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : CommSemigroup β
    inst✝¹ : LE β
    inst✝ : SubmultiplicativeHomClass F α β
    f : F
    a b : α
    ⊢ LE.le (f a) (HMul.hMul (f b) (f (HDiv.hDiv a b)))
  -/
  simpa only [mul_comm, div_mul_cancel] using map_mul_le_mul f (a / b) b
  /-
    🎉 no goals
  -/


@[to_additive existing]
theorem le_map_add_map_div [Group α] [AddCommSemigroup β] [LE β] [MulLEAddHomClass F α β] (f : F)
    (a b : α) : f a ≤ f b + f (a / b) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : AddCommSemigroup β
    inst✝¹ : LE β
    inst✝ : MulLEAddHomClass F α β
    f : F
    a b : α
    ⊢ LE.le (f a) (HAdd.hAdd (f b) (f (HDiv.hDiv a b)))
  -/
  simpa only [add_comm, div_mul_cancel] using map_mul_le_add f (a / b) b
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_map_div_mul_map_div [Group α] [CommSemigroup β] [LE β] [SubmultiplicativeHomClass F α β]
    (f : F) (a b c : α) : f (a / c) ≤ f (a / b) * f (b / c) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : CommSemigroup β
    inst✝¹ : LE β
    inst✝ : SubmultiplicativeHomClass F α β
    f : F
    a b c : α
    ⊢ LE.le (f (HDiv.hDiv a c)) (HMul.hMul (f (HDiv.hDiv a b)) (f (HDiv.hDiv b c)))
  -/
  simpa only [div_mul_div_cancel] using map_mul_le_mul f (a / b) (b / c)
  /-
    🎉 no goals
  -/


@[to_additive existing]
theorem le_map_div_add_map_div [Group α] [AddCommSemigroup β] [LE β] [MulLEAddHomClass F α β]
    (f : F) (a b c : α) : f (a / c) ≤ f (a / b) + f (b / c) := by
    /-
      F : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁴ : FunLike F α β
      inst✝³ : Group α
      inst✝² : AddCommSemigroup β
      inst✝¹ : LE β
      inst✝ : MulLEAddHomClass F α β
      f : F
      a b c : α
      ⊢ LE.le (f (HDiv.hDiv a c)) (HAdd.hAdd (f (HDiv.hDiv a b)) (f (HDiv.hDiv b c)))
    -/
    simpa only [div_mul_div_cancel] using map_mul_le_add f (a / b) (b / c)
    /-
      🎉 no goals
    -/


/-- Extension for the `positivity` tactic: nonnegative maps take nonnegative values. -/
@[positivity DFunLike.coe _ _]
def evalMap : PositivityExt where eval {_ β} _ _ e := do
  let .app (.app _ f) a ← whnfR e
    | throwError "not ↑f · where f is of NonnegHomClass"
  let pa ← mkAppOptM ``apply_nonneg #[none, none, β, none, none, none, none, f, a]
  pure (.nonnegative pa)


/-- `AddGroupSeminormClass F α` states that `F` is a type of `β`-valued seminorms on the additive
group `α`.

You should extend this class when you extend `AddGroupSeminorm`. -/
class AddGroupSeminormClass (F : Type*) (α β : outParam Type*)
    [AddGroup α] [OrderedAddCommMonoid β] [FunLike F α β]
  extends SubadditiveHomClass F α β : Prop where
  /-- The image of zero is zero. -/
  map_zero (f : F) : f 0 = 0
  /-- The map is invariant under negation of its argument. -/
  map_neg_eq_map (f : F) (a : α) : f (-a) = f a


/-- `GroupSeminormClass F α` states that `F` is a type of `β`-valued seminorms on the group `α`.

You should extend this class when you extend `GroupSeminorm`. -/
@[to_additive]
class GroupSeminormClass (F : Type*) (α β : outParam Type*)
    [Group α] [OrderedAddCommMonoid β] [FunLike F α β]
  extends MulLEAddHomClass F α β : Prop where
  /-- The image of one is zero. -/
  map_one_eq_zero (f : F) : f 1 = 0
  /-- The map is invariant under inversion of its argument. -/
  map_inv_eq_map (f : F) (a : α) : f a⁻¹ = f a


/-- `AddGroupNormClass F α` states that `F` is a type of `β`-valued norms on the additive group
`α`.

You should extend this class when you extend `AddGroupNorm`. -/
class AddGroupNormClass (F : Type*) (α β : outParam Type*)
    [AddGroup α] [OrderedAddCommMonoid β] [FunLike F α β]
  extends AddGroupSeminormClass F α β : Prop where
  /-- The argument is zero if its image under the map is zero. -/
  eq_zero_of_map_eq_zero (f : F) {a : α} : f a = 0 → a = 0


/-- `GroupNormClass F α` states that `F` is a type of `β`-valued norms on the group `α`.

You should extend this class when you extend `GroupNorm`. -/
@[to_additive]
class GroupNormClass (F : Type*) (α β : outParam Type*)
    [Group α] [OrderedAddCommMonoid β] [FunLike F α β]
  extends GroupSeminormClass F α β : Prop where
  /-- The argument is one if its image under the map is zero. -/
  eq_one_of_map_eq_zero (f : F) {a : α} : f a = 0 → a = 1


attribute [to_additive] GroupSeminormClass.toMulLEAddHomClass

-- See note [lower instance priority]

instance (priority := 100) AddGroupSeminormClass.toZeroHomClass [AddGroup α]
    [OrderedAddCommMonoid β] [AddGroupSeminormClass F α β] : ZeroHomClass F α β :=
  { ‹AddGroupSeminormClass F α β› with }


@[to_additive]
theorem map_div_le_add : f (x / y) ≤ f x + f y := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : FunLike F α β
    inst✝² : Group α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : GroupSeminormClass F α β
    f : F
    x y : α
    ⊢ LE.le (f (HDiv.hDiv x y)) (HAdd.hAdd (f x) (f y))
  -/
  rw [div_eq_mul_inv, ← map_inv_eq_map f y]
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : FunLike F α β
    inst✝² : Group α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : GroupSeminormClass F α β
    f : F
    x y : α
    ⊢ LE.le (f (HMul.hMul x (Inv.inv y))) (HAdd.hAdd (f x) (f (Inv.inv y)))
  -/
  exact map_mul_le_add _ _ _
  /-
    🎉 no goals
  -/


@[to_additive]
                                                  /-
                                                    F : Type u_2
                                                    α : Type u_3
                                                    β : Type u_4
                                                    inst✝³ : FunLike F α β
                                                    inst✝² : Group α
                                                    inst✝¹ : OrderedAddCommMonoid β
                                                    inst✝ : GroupSeminormClass F α β
                                                    f : F
                                                    x y : α
                                                    ⊢ Eq (f (HDiv.hDiv x y)) (f (HDiv.hDiv y x))
                                                  -/
theorem map_div_rev : f (x / y) = f (y / x) := by rw [← inv_div, map_inv_eq_map]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive]
theorem le_map_add_map_div' : f x ≤ f y + f (y / x) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : FunLike F α β
    inst✝² : Group α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : GroupSeminormClass F α β
    f : F
    x y : α
    ⊢ LE.le (f x) (HAdd.hAdd (f y) (f (HDiv.hDiv y x)))
  -/
  simpa only [add_comm, map_div_rev, div_mul_cancel] using map_mul_le_add f (x / y) y
  /-
    🎉 no goals
  -/


@[to_additive]
theorem abs_sub_map_le_div [Group α] [LinearOrderedAddCommGroup β] [GroupSeminormClass F α β]
    (f : F) (x y : α) : |f x - f y| ≤ f (x / y) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : FunLike F α β
    inst✝² : Group α
    inst✝¹ : LinearOrderedAddCommGroup β
    inst✝ : GroupSeminormClass F α β
    f : F
    x y : α
    ⊢ LE.le (abs (HSub.hSub (f x) (f y))) (f (HDiv.hDiv x y))
  -/
  rw [abs_sub_le_iff, sub_le_iff_le_add', sub_le_iff_le_add']
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : FunLike F α β
    inst✝² : Group α
    inst✝¹ : LinearOrderedAddCommGroup β
    inst✝ : GroupSeminormClass F α β
    f : F
    x y : α
    ⊢ And (LE.le (f x) (HAdd.hAdd (f y) (f (HDiv.hDiv x y)))) (LE.le (f y) (HAdd.h …
  -/
  exact ⟨le_map_add_map_div _ _ _, le_map_add_map_div' _ _ _⟩
  /-
    🎉 no goals
  -/

-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) GroupSeminormClass.toNonnegHomClass [Group α]
    [LinearOrderedAddCommMonoid β] [GroupSeminormClass F α β] : NonnegHomClass F α β :=
  { ‹GroupSeminormClass F α β› with
    apply_nonneg := fun f a =>
      (nsmul_nonneg_iff two_ne_zero).1 <| by
        /-
          ι : Type u_1
          F : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          δ : Type u_6
          inst✝³ : FunLike F α β
          inst✝² : Group α
          inst✝¹ : LinearOrderedAddCommMonoid β
          inst✝ : GroupSeminormClass F α β
          f : F
          a : α
          ⊢ LE.le 0 (HSMul.hSMul 2 (f a))
        -/
        rw [two_nsmul, ← map_one_eq_zero f, ← div_self' a]
        /-
          ι : Type u_1
          F : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          δ : Type u_6
          inst✝³ : FunLike F α β
          inst✝² : Group α
          inst✝¹ : LinearOrderedAddCommMonoid β
          inst✝ : GroupSeminormClass F α β
          f : F
          a : α
          ⊢ LE.le (f (HDiv.hDiv a a)) (HAdd.hAdd (f a) (f a))
        -/
        exact map_div_le_add _ _ _ }
        /-
          🎉 no goals
        -/


@[to_additive (attr := simp)]
theorem map_eq_zero_iff_eq_one : f x = 0 ↔ x = 1 :=
  ⟨eq_one_of_map_eq_zero _, by
    /-
      F : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝³ : FunLike F α β
      inst✝² : Group α
      inst✝¹ : OrderedAddCommMonoid β
      inst✝ : GroupNormClass F α β
      f : F
      x : α
      ⊢ Eq x 1 → Eq (f x) 0
    -/
    rintro rfl
    /-
      F : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝³ : FunLike F α β
      inst✝² : Group α
      inst✝¹ : OrderedAddCommMonoid β
      inst✝ : GroupNormClass F α β
      f : F
      ⊢ Eq (f 1) 0
    -/
    exact map_one_eq_zero _⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem map_ne_zero_iff_ne_one : f x ≠ 0 ↔ x ≠ 1 :=
  (map_eq_zero_iff_eq_one _).not


@[to_additive]
theorem map_pos_of_ne_one [Group α] [LinearOrderedAddCommMonoid β] [GroupNormClass F α β] (f : F)
    {x : α} (hx : x ≠ 1) : 0 < f x :=
  (apply_nonneg _ _).lt_of_ne <| ((map_ne_zero_iff_ne_one _).2 hx).symm


/-- `RingSeminormClass F α` states that `F` is a type of `β`-valued seminorms on the ring `α`.

You should extend this class when you extend `RingSeminorm`. -/
class RingSeminormClass (F : Type*) (α β : outParam Type*)
    [NonUnitalNonAssocRing α] [OrderedSemiring β] [FunLike F α β]
  extends AddGroupSeminormClass F α β, SubmultiplicativeHomClass F α β : Prop


/-- `RingNormClass F α` states that `F` is a type of `β`-valued norms on the ring `α`.

You should extend this class when you extend `RingNorm`. -/
class RingNormClass (F : Type*) (α β : outParam Type*)
    [NonUnitalNonAssocRing α] [OrderedSemiring β] [FunLike F α β]
  extends RingSeminormClass F α β, AddGroupNormClass F α β : Prop


/-- `MulRingSeminormClass F α` states that `F` is a type of `β`-valued multiplicative seminorms
on the ring `α`.

You should extend this class when you extend `MulRingSeminorm`. -/
class MulRingSeminormClass (F : Type*) (α β : outParam Type*)
    [NonAssocRing α] [OrderedSemiring β] [FunLike F α β]
  extends AddGroupSeminormClass F α β, MonoidWithZeroHomClass F α β : Prop

-- Lower the priority of these instances since they require synthesizing an order structure.

/-- `MulRingNormClass F α` states that `F` is a type of `β`-valued multiplicative norms on the
ring `α`.

You should extend this class when you extend `MulRingNorm`. -/
class MulRingNormClass (F : Type*) (α β : outParam Type*)
    [NonAssocRing α] [OrderedSemiring β] [FunLike F α β]
  extends MulRingSeminormClass F α β, AddGroupNormClass F α β : Prop

-- See note [out-param inheritance]
-- See note [lower instance priority]

instance (priority := 100) RingSeminormClass.toNonnegHomClass [NonUnitalNonAssocRing α]
    [LinearOrderedSemiring β] [RingSeminormClass F α β] : NonnegHomClass F α β :=
  AddGroupSeminormClass.toNonnegHomClass

-- See note [lower instance priority]

instance (priority := 100) MulRingSeminormClass.toRingSeminormClass [NonAssocRing α]
    [OrderedSemiring β] [MulRingSeminormClass F α β] : RingSeminormClass F α β :=
  { ‹MulRingSeminormClass F α β› with map_mul_le_mul := fun _ _ _ => (map_mul _ _ _).le }

-- See note [lower instance priority]

instance (priority := 100) MulRingNormClass.toRingNormClass [NonAssocRing α]
    [OrderedSemiring β] [MulRingNormClass F α β] : RingNormClass F α β :=
  { ‹MulRingNormClass F α β›, MulRingSeminormClass.toRingSeminormClass with }

