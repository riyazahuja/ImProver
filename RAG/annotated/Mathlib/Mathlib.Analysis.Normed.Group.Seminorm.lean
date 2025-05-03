/-- A seminorm on an additive group `G` is a function `f : G → ℝ` that preserves zero, is
subadditive and such that `f (-x) = f x` for all `x`. -/
structure AddGroupSeminorm (G : Type*) [AddGroup G] where
  -- Porting note: can't extend `ZeroHom G ℝ` because otherwise `to_additive` won't work since
  -- we aren't using old structures
  /-- The bare function of an `AddGroupSeminorm`. -/
  protected toFun : G → ℝ
  /-- The image of zero is zero. -/
  protected map_zero' : toFun 0 = 0
  /-- The seminorm is subadditive. -/
  protected add_le' : ∀ r s, toFun (r + s) ≤ toFun r + toFun s
  /-- The seminorm is invariant under negation. -/
  protected neg' : ∀ r, toFun (-r) = toFun r


/-- A seminorm on a group `G` is a function `f : G → ℝ` that sends one to zero, is submultiplicative
and such that `f x⁻¹ = f x` for all `x`. -/
@[to_additive]
structure GroupSeminorm (G : Type*) [Group G] where
  /-- The bare function of a `GroupSeminorm`. -/
  protected toFun : G → ℝ
  /-- The image of one is zero. -/
  protected map_one' : toFun 1 = 0
  /-- The seminorm applied to a product is dominated by the sum of the seminorm applied to the
  factors. -/
  protected mul_le' : ∀ x y, toFun (x * y) ≤ toFun x + toFun y
  /-- The seminorm is invariant under inversion. -/
  protected inv' : ∀ x, toFun x⁻¹ = toFun x


/-- A nonarchimedean seminorm on an additive group `G` is a function `f : G → ℝ` that preserves
zero, is nonarchimedean and such that `f (-x) = f x` for all `x`. -/
structure NonarchAddGroupSeminorm (G : Type*) [AddGroup G] extends ZeroHom G ℝ where
  /-- The seminorm applied to a sum is dominated by the maximum of the function applied to the
  addends. -/
  protected add_le_max' : ∀ r s, toFun (r + s) ≤ max (toFun r) (toFun s)
  /-- The seminorm is invariant under negation. -/
  protected neg' : ∀ r, toFun (-r) = toFun r


/-- A norm on an additive group `G` is a function `f : G → ℝ` that preserves zero, is subadditive
and such that `f (-x) = f x` and `f x = 0 → x = 0` for all `x`. -/
structure AddGroupNorm (G : Type*) [AddGroup G] extends AddGroupSeminorm G where
  /-- If the image under the seminorm is zero, then the argument is zero. -/
  protected eq_zero_of_map_eq_zero' : ∀ x, toFun x = 0 → x = 0


/-- A seminorm on a group `G` is a function `f : G → ℝ` that sends one to zero, is submultiplicative
and such that `f x⁻¹ = f x` and `f x = 0 → x = 1` for all `x`. -/
@[to_additive]
structure GroupNorm (G : Type*) [Group G] extends GroupSeminorm G where
  /-- If the image under the norm is zero, then the argument is one. -/
  protected eq_one_of_map_eq_zero' : ∀ x, toFun x = 0 → x = 1


/-- A nonarchimedean norm on an additive group `G` is a function `f : G → ℝ` that preserves zero, is
nonarchimedean and such that `f (-x) = f x` and `f x = 0 → x = 0` for all `x`. -/
structure NonarchAddGroupNorm (G : Type*) [AddGroup G] extends NonarchAddGroupSeminorm G where
  /-- If the image under the norm is zero, then the argument is zero. -/
  protected eq_zero_of_map_eq_zero' : ∀ x, toFun x = 0 → x = 0


/-- `NonarchAddGroupSeminormClass F α` states that `F` is a type of nonarchimedean seminorms on
the additive group `α`.

You should extend this class when you extend `NonarchAddGroupSeminorm`. -/
class NonarchAddGroupSeminormClass (F : Type*) (α : outParam Type*) [AddGroup α] [FunLike F α ℝ]
    extends NonarchimedeanHomClass F α ℝ : Prop where
  /-- The image of zero is zero. -/
  protected map_zero (f : F) : f 0 = 0
  /-- The seminorm is invariant under negation. -/
  protected map_neg_eq_map' (f : F) (a : α) : f (-a) = f a


/-- `NonarchAddGroupNormClass F α` states that `F` is a type of nonarchimedean norms on the
additive group `α`.

You should extend this class when you extend `NonarchAddGroupNorm`. -/
class NonarchAddGroupNormClass (F : Type*) (α : outParam Type*) [AddGroup α] [FunLike F α ℝ]
    extends NonarchAddGroupSeminormClass F α : Prop where
  /-- If the image under the norm is zero, then the argument is zero. -/
  protected eq_zero_of_map_eq_zero (f : F) {a : α} : f a = 0 → a = 0


theorem map_sub_le_max : f (x - y) ≤ max (f x) (f y) := by
  /-
    E : Type u_3
    F : Type u_4
    inst✝² : AddGroup E
    inst✝¹ : FunLike F E Real
    inst✝ : NonarchAddGroupSeminormClass F E
    f : F
    x y : E
    ⊢ LE.le (f (HSub.hSub x y)) (Max.max (f x) (f y))
  -/
  rw [sub_eq_add_neg, ← NonarchAddGroupSeminormClass.map_neg_eq_map' f y]
  /-
    E : Type u_3
    F : Type u_4
    inst✝² : AddGroup E
    inst✝¹ : FunLike F E Real
    inst✝ : NonarchAddGroupSeminormClass F E
    f : F
    x y : E
    ⊢ LE.le (f (HAdd.hAdd x (Neg.neg y))) (Max.max (f x) (f (Neg.neg y)))
  -/
  exact map_add_le_max _ _ _
  /-
    🎉 no goals
  -/


instance (priority := 100) NonarchAddGroupSeminormClass.toAddGroupSeminormClass
    [FunLike F E ℝ] [AddGroup E] [NonarchAddGroupSeminormClass F E] : AddGroupSeminormClass F E ℝ :=
  { ‹NonarchAddGroupSeminormClass F E› with
    map_add_le_add := fun f _ _ =>
      haveI h_nonneg : ∀ a, 0 ≤ f a := by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝² : FunLike F E Real
          inst✝¹ : AddGroup E
          inst✝ : NonarchAddGroupSeminormClass F E
          f : F
          x✝¹ x✝ : E
          ⊢ ∀ (a : E), LE.le 0 (f a)
        -/
        intro a
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝² : FunLike F E Real
          inst✝¹ : AddGroup E
          inst✝ : NonarchAddGroupSeminormClass F E
          f : F
          x✝¹ x✝ a : E
          ⊢ LE.le 0 (f a)
        -/
        rw [← NonarchAddGroupSeminormClass.map_zero f, ← sub_self a]
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝² : FunLike F E Real
          inst✝¹ : AddGroup E
          inst✝ : NonarchAddGroupSeminormClass F E
          f : F
          x✝¹ x✝ a : E
          ⊢ LE.le (f (HSub.hSub a a)) (f a)
        -/
        exact le_trans (map_sub_le_max _ _ _) (by rw [max_self (f a)])
        /-
          🎉 no goals
        -/
      le_trans (map_add_le_max _ _ _)
        (max_le (le_add_of_nonneg_right (h_nonneg _)) (le_add_of_nonneg_left (h_nonneg _)))
    map_neg_eq_map := NonarchAddGroupSeminormClass.map_neg_eq_map' }

-- See note [lower instance priority]

instance (priority := 100) NonarchAddGroupNormClass.toAddGroupNormClass
    [FunLike F E ℝ] [AddGroup E] [NonarchAddGroupNormClass F E] : AddGroupNormClass F E ℝ :=
  { ‹NonarchAddGroupNormClass F E› with
    map_add_le_add := map_add_le_add
    map_neg_eq_map := NonarchAddGroupSeminormClass.map_neg_eq_map' }


@[to_additive]
instance funLike : FunLike (GroupSeminorm E) E ℝ where
  coe f := f.toFun
                             /-
                               R : Type u_1
                               R' : Type u_2
                               E : Type u_3
                               F : Type u_4
                               G : Type u_5
                               inst✝² : Group E
                               inst✝¹ : Group F
                               inst✝ : Group G
                               p q f g : GroupSeminorm E
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
instance groupSeminormClass : GroupSeminormClass (GroupSeminorm E) E ℝ where
  map_one_eq_zero f := f.map_one'
  map_mul_le_add f := f.mul_le'
  map_inv_eq_map f := f.inv'


@[to_additive (attr := simp)]
theorem toFun_eq_coe : p.toFun = p :=
  rfl


@[to_additive (attr := ext)]
theorem ext : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


@[to_additive]
instance : PartialOrder (GroupSeminorm E) :=
  PartialOrder.lift _ DFunLike.coe_injective


@[to_additive]
theorem le_def : p ≤ q ↔ (p : E → ℝ) ≤ q :=
  Iff.rfl


@[to_additive]
theorem lt_def : p < q ↔ (p : E → ℝ) < q :=
  Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_le_coe : (p : E → ℝ) ≤ q ↔ p ≤ q :=
  Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_lt_coe : (p : E → ℝ) < q ↔ p < q :=
  Iff.rfl


@[to_additive]
instance instZeroGroupSeminorm : Zero (GroupSeminorm E) :=
  ⟨{  toFun := 0
      map_one' := Pi.zero_apply _
      mul_le' := fun _ _ => (zero_add _).ge
      inv' := fun _ => rfl }⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_zero : ⇑(0 : GroupSeminorm E) = 0 :=
  rfl


@[to_additive (attr := simp)]
theorem zero_apply (x : E) : (0 : GroupSeminorm E) x = 0 :=
  rfl


@[to_additive]
instance : Inhabited (GroupSeminorm E) :=
  ⟨0⟩


@[to_additive]
instance : Add (GroupSeminorm E) :=
  ⟨fun p q =>
    { toFun := fun x => p x + q x
                     /-
                       R : Type u_1
                       R' : Type u_2
                       E : Type u_3
                       F : Type u_4
                       G : Type u_5
                       inst✝² : Group E
                       inst✝¹ : Group F
                       inst✝ : Group G
                       p✝ q✝ : GroupSeminorm E
                       f : MonoidHom F E
                       p q : GroupSeminorm E
                       ⊢ Eq ((fun x => HAdd.hAdd (p x) (q x)) 1) 0
                     -/
      map_one' := by simp_rw [map_one_eq_zero p, map_one_eq_zero q, zero_add]
                     /-
                       🎉 no goals
                     -/
      mul_le' := fun _ _ =>
        (add_le_add (map_mul_le_add p _ _) <| map_mul_le_add q _ _).trans_eq <|
          add_add_add_comm _ _ _ _
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝² : Group E
                            inst✝¹ : Group F
                            inst✝ : Group G
                            p✝ q✝ : GroupSeminorm E
                            f : MonoidHom F E
                            p q : GroupSeminorm E
                            x : E
                            ⊢ Eq ((fun x => HAdd.hAdd (p x) (q x)) (Inv.inv x)) ((fun x => HAdd.hAdd (p x) …
                          -/
      inv' := fun x => by simp_rw [map_inv_eq_map p, map_inv_eq_map q] }⟩
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp)]
theorem coe_add : ⇑(p + q) = p + q :=
  rfl


@[to_additive (attr := simp)]
theorem add_apply (x : E) : (p + q) x = p x + q x :=
  rfl

-- TODO: define `SupSet` too, from the skeleton at
-- https://github.com/leanprover-community/mathlib/pull/11329#issuecomment-1008915345

@[to_additive]
instance : Max (GroupSeminorm E) :=
  ⟨fun p q =>
    { toFun := p ⊔ q
      map_one' := by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝² : Group E
          inst✝¹ : Group F
          inst✝ : Group G
          p✝ q✝ : GroupSeminorm E
          f : MonoidHom F E
          p q : GroupSeminorm E
          ⊢ Eq (Max.max (⇑p) (⇑q) 1) 0
        -/
        rw [Pi.sup_apply, ← map_one_eq_zero p, sup_eq_left, map_one_eq_zero p, map_one_eq_zero q]
        /-
          🎉 no goals
        -/
      mul_le' := fun x y =>
        sup_le ((map_mul_le_add p x y).trans <| add_le_add le_sup_left le_sup_left)
          ((map_mul_le_add q x y).trans <| add_le_add le_sup_right le_sup_right)
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝² : Group E
                            inst✝¹ : Group F
                            inst✝ : Group G
                            p✝ q✝ : GroupSeminorm E
                            f : MonoidHom F E
                            p q : GroupSeminorm E
                            x : E
                            ⊢ Eq (Max.max (⇑p) (⇑q) (Inv.inv x)) (Max.max (⇑p) (⇑q) x)
                          -/
      inv' := fun x => by rw [Pi.sup_apply, Pi.sup_apply, map_inv_eq_map p, map_inv_eq_map q] }⟩
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp, norm_cast)]
theorem coe_sup : ⇑(p ⊔ q) = ⇑p ⊔ ⇑q :=
  rfl


@[to_additive (attr := simp)]
theorem sup_apply (x : E) : (p ⊔ q) x = p x ⊔ q x :=
  rfl


@[to_additive]
instance semilatticeSup : SemilatticeSup (GroupSeminorm E) :=
  DFunLike.coe_injective.semilatticeSup _ coe_sup


/-- Composition of a group seminorm with a monoid homomorphism as a group seminorm. -/
@[to_additive "Composition of an additive group seminorm with an additive monoid homomorphism as an
additive group seminorm."]
def comp (p : GroupSeminorm E) (f : F →* E) : GroupSeminorm F where
  toFun x := p (f x)
                 /-
                   R : Type u_1
                   R' : Type u_2
                   E : Type u_3
                   F : Type u_4
                   G : Type u_5
                   inst✝² : Group E
                   inst✝¹ : Group F
                   inst✝ : Group G
                   p✝ q : GroupSeminorm E
                   f✝ : MonoidHom F E
                   p : GroupSeminorm E
                   f : MonoidHom F E
                   ⊢ Eq ((fun x => p (f x)) 1) 0
                 -/
  map_one' := by simp_rw [f.map_one, map_one_eq_zero p]
                 /-
                   🎉 no goals
                 -/
  mul_le' _ _ := (congr_arg p <| f.map_mul _ _).trans_le <| map_mul_le_add p _ _
               /-
                 R : Type u_1
                 R' : Type u_2
                 E : Type u_3
                 F : Type u_4
                 G : Type u_5
                 inst✝² : Group E
                 inst✝¹ : Group F
                 inst✝ : Group G
                 p✝ q : GroupSeminorm E
                 f✝ : MonoidHom F E
                 p : GroupSeminorm E
                 f : MonoidHom F E
                 x : F
                 ⊢ Eq ((fun x => p (f x)) (Inv.inv x)) ((fun x => p (f x)) x)
               -/
  inv' x := by simp_rw [map_inv, map_inv_eq_map p]
               /-
                 🎉 no goals
               -/


@[to_additive (attr := simp)]
theorem coe_comp : ⇑(p.comp f) = p ∘ f :=
  rfl


@[to_additive (attr := simp)]
theorem comp_apply (x : F) : (p.comp f) x = p (f x) :=
  rfl


@[to_additive (attr := simp)]
theorem comp_id : p.comp (MonoidHom.id _) = p :=
  ext fun _ => rfl


@[to_additive (attr := simp)]
theorem comp_zero : p.comp (1 : F →* E) = 0 :=
  ext fun _ => map_one_eq_zero p


@[to_additive (attr := simp)]
theorem zero_comp : (0 : GroupSeminorm E).comp f = 0 :=
  ext fun _ => rfl


@[to_additive]
theorem comp_assoc (g : F →* E) (f : G →* F) : p.comp (g.comp f) = (p.comp g).comp f :=
  ext fun _ => rfl


@[to_additive]
theorem add_comp (f : F →* E) : (p + q).comp f = p.comp f + q.comp f :=
  ext fun _ => rfl


@[to_additive]
theorem comp_mono (hp : p ≤ q) : p.comp f ≤ q.comp f := fun _ => hp _


@[to_additive]
theorem comp_mul_le (f g : F →* E) : p.comp (f * g) ≤ p.comp f + p.comp g := fun _ =>
  map_mul_le_add p _ _


@[to_additive]
theorem mul_bddBelow_range_add {p q : GroupSeminorm E} {x : E} :
    BddBelow (range fun y => p y + q (x / y)) :=
  ⟨0, by
    /-
      E : Type u_3
      inst✝ : CommGroup E
      p q : GroupSeminorm E
      x : E
      ⊢ Membership.mem (lowerBounds (Set.range fun y => HAdd.hAdd (p y) (q (HDiv.hDi …
    -/
    rintro _ ⟨x, rfl⟩
    /-
      case intro
      E : Type u_3
      inst✝ : CommGroup E
      p q : GroupSeminorm E
      x✝ x : E
      ⊢ LE.le 0 ((fun y => HAdd.hAdd (p y) (q (HDiv.hDiv x✝ y))) x)
    -/
    dsimp
    /-
      case intro
      E : Type u_3
      inst✝ : CommGroup E
      p q : GroupSeminorm E
      x✝ x : E
      ⊢ LE.le 0 (HAdd.hAdd (p x) (q (HDiv.hDiv x✝ x)))
    -/
    positivity⟩
    /-
      🎉 no goals
    -/


@[to_additive]
noncomputable instance : Min (GroupSeminorm E) :=
  ⟨fun p q =>
    { toFun := fun x => ⨅ y, p y + q (x / y)
      map_one' :=
        ciInf_eq_of_forall_ge_of_forall_gt_exists_lt
          -- Porting note: replace `add_nonneg` with `positivity` once we have the extension
          (fun _ => add_nonneg (apply_nonneg _ _) (apply_nonneg _ _)) fun r hr =>
                 /-
                   R : Type u_1
                   R' : Type u_2
                   E : Type u_3
                   F : Type u_4
                   G : Type u_5
                   inst✝¹ : CommGroup E
                   inst✝ : CommGroup F
                   p✝ q✝ : GroupSeminorm E
                   x : E
                   p q : GroupSeminorm E
                   r : Real
                   hr : LT.lt 0 r
                   ⊢ LT.lt (HAdd.hAdd (p 1) (q (1 / 1))) r
                 -/
          ⟨1, by rwa [div_one, map_one_eq_zero p, map_one_eq_zero q, add_zero]⟩
                 /-
                   🎉 no goals
                 -/
      mul_le' := fun x y =>
        le_ciInf_add_ciInf fun u v => by
          /-
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝¹ : CommGroup E
            inst✝ : CommGroup F
            p✝ q✝ : GroupSeminorm E
            x✝ : E
            p q : GroupSeminorm E
            x y u v : E
            ⊢ LE.le ((fun x => iInf fun y => HAdd.hAdd (p y) (q (HDiv.hDiv x y))) (HMul.hM …
          -/
          refine ciInf_le_of_le mul_bddBelow_range_add (u * v) ?_
          /-
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝¹ : CommGroup E
            inst✝ : CommGroup F
            p✝ q✝ : GroupSeminorm E
            x✝ : E
            p q : GroupSeminorm E
            x y u v : E
            ⊢ LE.le (HAdd.hAdd (p (HMul.hMul u v)) (q (HDiv.hDiv (HMul.hMul x y) (HMul.hMu …
          -/
          rw [mul_div_mul_comm, add_add_add_comm]
          /-
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝¹ : CommGroup E
            inst✝ : CommGroup F
            p✝ q✝ : GroupSeminorm E
            x✝ : E
            p q : GroupSeminorm E
            x y u v : E
            ⊢ LE.le (HAdd.hAdd (p (HMul.hMul u v)) (q (HMul.hMul (HDiv.hDiv x u) (HDiv.hDi …
          -/
          exact add_le_add (map_mul_le_add p _ _) (map_mul_le_add q _ _)
          /-
            🎉 no goals
          -/
      inv' := fun x =>
        (inv_surjective.iInf_comp _).symm.trans <| by
          /-
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝¹ : CommGroup E
            inst✝ : CommGroup F
            p✝ q✝ : GroupSeminorm E
            x✝ : E
            p q : GroupSeminorm E
            x : E
            ⊢ Eq (iInf fun x_1 => HAdd.hAdd (p (Inv.inv x_1)) (q (HDiv.hDiv (Inv.inv x) (I …
          -/
          simp_rw [map_inv_eq_map p, ← inv_div', map_inv_eq_map q] }⟩
          /-
            🎉 no goals
          -/


@[to_additive (attr := simp)]
theorem inf_apply : (p ⊓ q) x = ⨅ y, p y + q (x / y) :=
  rfl


@[to_additive]
noncomputable instance : Lattice (GroupSeminorm E) :=
  { GroupSeminorm.semilatticeSup with
    inf := (· ⊓ ·)
    inf_le_left := fun p q x =>
                                                    /-
                                                      R : Type u_1
                                                      R' : Type u_2
                                                      E : Type u_3
                                                      F : Type u_4
                                                      G : Type u_5
                                                      inst✝¹ : CommGroup E
                                                      inst✝ : CommGroup F
                                                      p✝ q✝ : GroupSeminorm E
                                                      x✝ : E
                                                      p q : GroupSeminorm E
                                                      x : E
                                                      ⊢ LE.le (HAdd.hAdd (p x) (q (HDiv.hDiv x x))) ((fun f => ⇑f) p x)
                                                    -/
      ciInf_le_of_le mul_bddBelow_range_add x <| by rw [div_self', map_one_eq_zero q, add_zero]
                                                    /-
                                                      🎉 no goals
                                                    -/
    inf_le_right := fun p q x =>
      ciInf_le_of_le mul_bddBelow_range_add (1 : E) <| by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝¹ : CommGroup E
          inst✝ : CommGroup F
          p✝ q✝ : GroupSeminorm E
          x✝ : E
          p q : GroupSeminorm E
          x : E
          ⊢ LE.le (HAdd.hAdd (p 1) (q (HDiv.hDiv x 1))) ((fun f => ⇑f) q x)
        -/
        simpa only [div_one x, map_one_eq_zero p, zero_add (q x)] using le_rfl
        /-
          🎉 no goals
        -/
    le_inf := fun a _ _ hb hc _ =>
      le_ciInf fun _ => (le_map_add_map_div a _ _).trans <| add_le_add (hb _) (hc _) }


instance toOne [DecidableEq E] : One (AddGroupSeminorm E) :=
  ⟨{  toFun := fun x => if x = 0 then 0 else 1
      map_zero' := if_pos rfl
      add_le' := fun x y => by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝⁴ : AddGroup E
          inst✝³ : SMul R Real
          inst✝² : SMul R NNReal
          inst✝¹ : IsScalarTower R NNReal Real
          inst✝ : DecidableEq E
          x y : E
          ⊢ LE.le ((fun x => ite (Eq x 0) 0 1) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => it …
        -/
        by_cases hx : x = 0
          /-
            case pos
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Eq x 0
            ⊢ LE.le ((fun x => ite (Eq x 0) 0 1) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => it …
          -/
        · simp only
          /-
            case pos
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Eq x 0
            ⊢ LE.le (ite (Eq (HAdd.hAdd x y) 0) 0 1) (HAdd.hAdd (ite (Eq x 0) 0 1) (ite (E …
          -/
          rw [if_pos hx, hx, zero_add, zero_add]
          /-
            🎉 no goals
          -/
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 0)
            ⊢ LE.le ((fun x => ite (Eq x 0) 0 1) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => it …
          -/
        · simp only
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 0)
            ⊢ LE.le (ite (Eq (HAdd.hAdd x y) 0) 0 1) (HAdd.hAdd (ite (Eq x 0) 0 1) (ite (E …
          -/
          rw [if_neg hx]
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 0)
            ⊢ LE.le (ite (Eq (HAdd.hAdd x y) 0) 0 1) (HAdd.hAdd 1 (ite (Eq y 0) 0 1))
          -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
          refine le_add_of_le_of_nonneg ?_ ?_ <;> split_ifs <;> norm_num
                                                                /-
                                                                  🎉 no goals
                                                                -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝⁴ : AddGroup E
                            inst✝³ : SMul R Real
                            inst✝² : SMul R NNReal
                            inst✝¹ : IsScalarTower R NNReal Real
                            inst✝ : DecidableEq E
                            x : E
                            ⊢ Eq ((fun x => ite (Eq x 0) 0 1) (Neg.neg x)) ((fun x => ite (Eq x 0) 0 1) x)
                          -/
      neg' := fun x => by simp_rw [neg_eq_zero] }⟩
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem apply_one [DecidableEq E] (x : E) : (1 : AddGroupSeminorm E) x = if x = 0 then 0 else 1 :=
  rfl


/-- Any action on `ℝ` which factors through `ℝ≥0` applies to an `AddGroupSeminorm`. -/
instance toSMul : SMul R (AddGroupSeminorm E) :=
  ⟨fun r p =>
    { toFun := fun x => r • p x
      map_zero' := by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : AddGroup E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : AddGroupSeminorm E
          ⊢ Eq ((fun x => HSMul.hSMul r (p x)) 0) 0
        -/
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul, map_zero, mul_zero]
        /-
          🎉 no goals
        -/
      add_le' := fun _ _ => by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : AddGroup E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : AddGroupSeminorm E
          x✝¹ x✝ : E
          ⊢ LE.le ((fun x => HSMul.hSMul r (p x)) (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd ((fun x …
        -/
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul, ← mul_add]
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : AddGroup E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : AddGroupSeminorm E
          x✝¹ x✝ : E
          ⊢ LE.le (HMul.hMul (↑(HSMul.hSMul r 1)) (p (HAdd.hAdd x✝¹ x✝))) (HMul.hMul (↑( …
        -/
        gcongr
        /-
          case h
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : AddGroup E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : AddGroupSeminorm E
          x✝¹ x✝ : E
          ⊢ LE.le (p (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (p x✝¹) (p x✝))
        -/
        apply map_add_le_add
        /-
          🎉 no goals
        -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝³ : AddGroup E
                            inst✝² : SMul R Real
                            inst✝¹ : SMul R NNReal
                            inst✝ : IsScalarTower R NNReal Real
                            r : R
                            p : AddGroupSeminorm E
                            x : E
                            ⊢ Eq ((fun x => HSMul.hSMul r (p x)) (Neg.neg x)) ((fun x => HSMul.hSMul r (p  …
                          -/
      neg' := fun x => by simp_rw [map_neg_eq_map] }⟩
                          /-
                            🎉 no goals
                          -/


@[simp, norm_cast]
theorem coe_smul (r : R) (p : AddGroupSeminorm E) : ⇑(r • p) = r • ⇑p :=
  rfl


@[simp]
theorem smul_apply (r : R) (p : AddGroupSeminorm E) (x : E) : (r • p) x = r • p x :=
  rfl


instance isScalarTower [SMul R' ℝ] [SMul R' ℝ≥0] [IsScalarTower R' ℝ≥0 ℝ] [SMul R R']
    [IsScalarTower R R' ℝ] : IsScalarTower R R' (AddGroupSeminorm E) :=
  ⟨fun r a p => ext fun x => smul_assoc r a (p x)⟩


theorem smul_sup (r : R) (p q : AddGroupSeminorm E) : r • (p ⊔ q) = r • p ⊔ r • q :=
  have Real.smul_max : ∀ x y : ℝ, r • max x y = max (r • x) (r • y) := fun x y => by
    simpa only [← smul_eq_mul, ← NNReal.smul_def, smul_one_smul ℝ≥0 r (_ : ℝ)] using
      mul_max_of_nonneg x y (r • (1 : ℝ≥0) : ℝ≥0).coe_nonneg
  ext fun _ => Real.smul_max _ _


instance funLike : FunLike (NonarchAddGroupSeminorm E) E ℝ where
  coe f := f.toFun
                             /-
                               R : Type u_1
                               R' : Type u_2
                               E : Type u_3
                               F : Type u_4
                               G : Type u_5
                               inst✝ : AddGroup E
                               p q f g : NonarchAddGroupSeminorm E
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨_, _⟩, _, _⟩ := f; cases g; congr
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance nonarchAddGroupSeminormClass :
    NonarchAddGroupSeminormClass (NonarchAddGroupSeminorm E) E where
  map_add_le_max f := f.add_le_max'
  map_zero f := f.map_zero'
  map_neg_eq_map' f := f.neg'

-- Porting note: `simpNF` said the left hand side simplified to this

@[simp]
theorem toZeroHom_eq_coe : ⇑p.toZeroHom = p := by
  /-
    E : Type u_3
    inst✝ : AddGroup E
    p : NonarchAddGroupSeminorm E
    ⊢ Eq ⇑p.toZeroHom ⇑p
  -/
  rfl
  /-
    🎉 no goals
  -/


@[ext]
theorem ext : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


noncomputable instance : PartialOrder (NonarchAddGroupSeminorm E) :=
  PartialOrder.lift _ DFunLike.coe_injective


theorem le_def : p ≤ q ↔ (p : E → ℝ) ≤ q :=
  Iff.rfl


theorem lt_def : p < q ↔ (p : E → ℝ) < q :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_le_coe : (p : E → ℝ) ≤ q ↔ p ≤ q :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_lt_coe : (p : E → ℝ) < q ↔ p < q :=
  Iff.rfl


instance : Zero (NonarchAddGroupSeminorm E) :=
  ⟨{  toFun := 0
      map_zero' := Pi.zero_apply _
                                   /-
                                     R : Type u_1
                                     R' : Type u_2
                                     E : Type u_3
                                     F : Type u_4
                                     G : Type u_5
                                     inst✝ : AddGroup E
                                     p q : NonarchAddGroupSeminorm E
                                     r s : E
                                     ⊢ LE.le ({ toFun := 0, map_zero' := ⋯ }.toFun (HAdd.hAdd r s)) (Max.max ({ toF …
                                   -/
      add_le_max' := fun r s => by simp only [Pi.zero_apply]; rw [max_eq_right]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
      neg' := fun _ => rfl }⟩


@[simp, norm_cast]
theorem coe_zero : ⇑(0 : NonarchAddGroupSeminorm E) = 0 :=
  rfl


@[simp]
theorem zero_apply (x : E) : (0 : NonarchAddGroupSeminorm E) x = 0 :=
  rfl


instance : Inhabited (NonarchAddGroupSeminorm E) :=
  ⟨0⟩

-- TODO: define `SupSet` too, from the skeleton at
-- https://github.com/leanprover-community/mathlib/pull/11329#issuecomment-1008915345

instance : Max (NonarchAddGroupSeminorm E) :=
  ⟨fun p q =>
    { toFun := p ⊔ q
                      /-
                        R : Type u_1
                        R' : Type u_2
                        E : Type u_3
                        F : Type u_4
                        G : Type u_5
                        inst✝ : AddGroup E
                        p✝ q✝ p q : NonarchAddGroupSeminorm E
                        ⊢ Eq (Max.max (⇑p) (⇑q) 0) 0
                      -/
      map_zero' := by rw [Pi.sup_apply, ← map_zero p, sup_eq_left, map_zero p, map_zero q]
                      /-
                        🎉 no goals
                      -/
      add_le_max' := fun x y =>
        sup_le ((map_add_le_max p x y).trans <| max_le_max le_sup_left le_sup_left)
          ((map_add_le_max q x y).trans <| max_le_max le_sup_right le_sup_right)
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝ : AddGroup E
                            p✝ q✝ p q : NonarchAddGroupSeminorm E
                            x : E
                            ⊢ Eq ({ toFun := Max.max ⇑p ⇑q, map_zero' := ⋯ }.toFun (Neg.neg x)) ({ toFun : …
                          -/
      neg' := fun x => by simp_rw [Pi.sup_apply, map_neg_eq_map p, map_neg_eq_map q]}⟩
                          /-
                            🎉 no goals
                          -/


@[simp, norm_cast]
theorem coe_sup : ⇑(p ⊔ q) = ⇑p ⊔ ⇑q :=
  rfl


@[simp]
theorem sup_apply (x : E) : (p ⊔ q) x = p x ⊔ q x :=
  rfl


noncomputable instance : SemilatticeSup (NonarchAddGroupSeminorm E) :=
  DFunLike.coe_injective.semilatticeSup _ coe_sup


theorem add_bddBelow_range_add {p q : NonarchAddGroupSeminorm E} {x : E} :
    BddBelow (range fun y => p y + q (x - y)) :=
  ⟨0, by
    /-
      E : Type u_3
      inst✝ : AddCommGroup E
      p q : NonarchAddGroupSeminorm E
      x : E
      ⊢ Membership.mem (lowerBounds (Set.range fun y => HAdd.hAdd (p y) (q (HSub.hSu …
    -/
    rintro _ ⟨x, rfl⟩
    /-
      case intro
      E : Type u_3
      inst✝ : AddCommGroup E
      p q : NonarchAddGroupSeminorm E
      x✝ x : E
      ⊢ LE.le 0 ((fun y => HAdd.hAdd (p y) (q (HSub.hSub x✝ y))) x)
    -/
    dsimp
    /-
      case intro
      E : Type u_3
      inst✝ : AddCommGroup E
      p q : NonarchAddGroupSeminorm E
      x✝ x : E
      ⊢ LE.le 0 (HAdd.hAdd (p x) (q (HSub.hSub x✝ x)))
    -/
    positivity⟩
    /-
      🎉 no goals
    -/


@[to_additive existing AddGroupSeminorm.toOne]
instance toOne [DecidableEq E] : One (GroupSeminorm E) :=
  ⟨{  toFun := fun x => if x = 1 then 0 else 1
      map_one' := if_pos rfl
      mul_le' := fun x y => by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝⁴ : Group E
          inst✝³ : SMul R Real
          inst✝² : SMul R NNReal
          inst✝¹ : IsScalarTower R NNReal Real
          inst✝ : DecidableEq E
          x y : E
          ⊢ LE.le ((fun x => ite (Eq x 1) 0 1) (HMul.hMul x y)) (HAdd.hAdd ((fun x => it …
        -/
        by_cases hx : x = 1
          /-
            case pos
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : Group E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Eq x 1
            ⊢ LE.le ((fun x => ite (Eq x 1) 0 1) (HMul.hMul x y)) (HAdd.hAdd ((fun x => it …
          -/
        · simp only
          /-
            case pos
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : Group E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Eq x 1
            ⊢ LE.le (ite (Eq (HMul.hMul x y) 1) 0 1) (HAdd.hAdd (ite (Eq x 1) 0 1) (ite (E …
          -/
          rw [if_pos hx, hx, one_mul, zero_add]
          /-
            🎉 no goals
          -/
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : Group E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 1)
            ⊢ LE.le ((fun x => ite (Eq x 1) 0 1) (HMul.hMul x y)) (HAdd.hAdd ((fun x => it …
          -/
        · simp only
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : Group E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 1)
            ⊢ LE.le (ite (Eq (HMul.hMul x y) 1) 0 1) (HAdd.hAdd (ite (Eq x 1) 0 1) (ite (E …
          -/
          rw [if_neg hx]
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : Group E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 1)
            ⊢ LE.le (ite (Eq (HMul.hMul x y) 1) 0 1) (HAdd.hAdd 1 (ite (Eq y 1) 0 1))
          -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
          refine le_add_of_le_of_nonneg ?_ ?_ <;> split_ifs <;> norm_num
                                                                /-
                                                                  🎉 no goals
                                                                -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝⁴ : Group E
                            inst✝³ : SMul R Real
                            inst✝² : SMul R NNReal
                            inst✝¹ : IsScalarTower R NNReal Real
                            inst✝ : DecidableEq E
                            x : E
                            ⊢ Eq ((fun x => ite (Eq x 1) 0 1) (Inv.inv x)) ((fun x => ite (Eq x 1) 0 1) x)
                          -/
      inv' := fun x => by simp_rw [inv_eq_one] }⟩
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp) existing AddGroupSeminorm.apply_one]
theorem apply_one [DecidableEq E] (x : E) : (1 : GroupSeminorm E) x = if x = 1 then 0 else 1 :=
  rfl


/-- Any action on `ℝ` which factors through `ℝ≥0` applies to an `AddGroupSeminorm`. -/
@[to_additive existing AddGroupSeminorm.toSMul]
instance : SMul R (GroupSeminorm E) :=
  ⟨fun r p =>
    { toFun := fun x => r • p x
      map_one' := by
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul, map_one_eq_zero p,
          mul_zero]
      mul_le' := fun _ _ => by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : Group E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : GroupSeminorm E
          x✝¹ x✝ : E
          ⊢ LE.le ((fun x => HSMul.hSMul r (p x)) (HMul.hMul x✝¹ x✝)) (HAdd.hAdd ((fun x …
        -/
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul, ← mul_add]
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : Group E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : GroupSeminorm E
          x✝¹ x✝ : E
          ⊢ LE.le (HMul.hMul (↑(HSMul.hSMul r 1)) (p (HMul.hMul x✝¹ x✝))) (HMul.hMul (↑( …
        -/
        gcongr
        /-
          case h
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : Group E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : GroupSeminorm E
          x✝¹ x✝ : E
          ⊢ LE.le (p (HMul.hMul x✝¹ x✝)) (HAdd.hAdd (p x✝¹) (p x✝))
        -/
        apply map_mul_le_add
        /-
          🎉 no goals
        -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝³ : Group E
                            inst✝² : SMul R Real
                            inst✝¹ : SMul R NNReal
                            inst✝ : IsScalarTower R NNReal Real
                            r : R
                            p : GroupSeminorm E
                            x : E
                            ⊢ Eq ((fun x => HSMul.hSMul r (p x)) (Inv.inv x)) ((fun x => HSMul.hSMul r (p  …
                          -/
      inv' := fun x => by simp_rw [map_inv_eq_map p] }⟩
                          /-
                            🎉 no goals
                          -/


@[to_additive existing AddGroupSeminorm.isScalarTower]
instance [SMul R' ℝ] [SMul R' ℝ≥0] [IsScalarTower R' ℝ≥0 ℝ] [SMul R R'] [IsScalarTower R R' ℝ] :
    IsScalarTower R R' (GroupSeminorm E) :=
  ⟨fun r a p => ext fun x => smul_assoc r a <| p x⟩


@[to_additive (attr := simp, norm_cast) existing AddGroupSeminorm.coe_smul]
theorem coe_smul (r : R) (p : GroupSeminorm E) : ⇑(r • p) = r • ⇑p :=
  rfl


@[to_additive (attr := simp) existing AddGroupSeminorm.smul_apply]
theorem smul_apply (r : R) (p : GroupSeminorm E) (x : E) : (r • p) x = r • p x :=
  rfl


@[to_additive existing AddGroupSeminorm.smul_sup]
theorem smul_sup (r : R) (p q : GroupSeminorm E) : r • (p ⊔ q) = r • p ⊔ r • q :=
  have Real.smul_max : ∀ x y : ℝ, r • max x y = max (r • x) (r • y) := fun x y => by
    simpa only [← smul_eq_mul, ← NNReal.smul_def, smul_one_smul ℝ≥0 r (_ : ℝ)] using
      mul_max_of_nonneg x y (r • (1 : ℝ≥0) : ℝ≥0).coe_nonneg
  ext fun _ => Real.smul_max _ _


instance [DecidableEq E] : One (NonarchAddGroupSeminorm E) :=
  ⟨{  toFun := fun x => if x = 0 then 0 else 1
      map_zero' := if_pos rfl
      add_le_max' := fun x y => by
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝⁴ : AddGroup E
          inst✝³ : SMul R Real
          inst✝² : SMul R NNReal
          inst✝¹ : IsScalarTower R NNReal Real
          inst✝ : DecidableEq E
          x y : E
          ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_zero' := ⋯ }.toFun (HAdd.hA …
        -/
        by_cases hx : x = 0
          /-
            case pos
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Eq x 0
            ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_zero' := ⋯ }.toFun (HAdd.hA …
          -/
        · simp_rw [if_pos hx, hx, zero_add]
          /-
            case pos
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Eq x 0
            ⊢ LE.le (ite (Eq y 0) 0 1) (Max.max 0 (ite (Eq y 0) 0 1))
          -/
          exact le_max_of_le_right (le_refl _)
          /-
            🎉 no goals
          -/
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 0)
            ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_zero' := ⋯ }.toFun (HAdd.hA …
          -/
        · simp_rw [if_neg hx]
          /-
            case neg
            R : Type u_1
            R' : Type u_2
            E : Type u_3
            F : Type u_4
            G : Type u_5
            inst✝⁴ : AddGroup E
            inst✝³ : SMul R Real
            inst✝² : SMul R NNReal
            inst✝¹ : IsScalarTower R NNReal Real
            inst✝ : DecidableEq E
            x y : E
            hx : Not (Eq x 0)
            ⊢ LE.le (ite (Eq (HAdd.hAdd x y) 0) 0 1) (Max.max 1 (ite (Eq y 0) 0 1))
          -/
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
          split_ifs <;> simp
                        /-
                          🎉 no goals
                        -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝⁴ : AddGroup E
                            inst✝³ : SMul R Real
                            inst✝² : SMul R NNReal
                            inst✝¹ : IsScalarTower R NNReal Real
                            inst✝ : DecidableEq E
                            x : E
                            ⊢ Eq ({ toFun := fun x => ite (Eq x 0) 0 1, map_zero' := ⋯ }.toFun (Neg.neg x) …
                          -/
      neg' := fun x => by simp_rw [neg_eq_zero] }⟩
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem apply_one [DecidableEq E] (x : E) :
    (1 : NonarchAddGroupSeminorm E) x = if x = 0 then 0 else 1 :=
  rfl


/-- Any action on `ℝ` which factors through `ℝ≥0` applies to a `NonarchAddGroupSeminorm`. -/
instance : SMul R (NonarchAddGroupSeminorm E) :=
  ⟨fun r p =>
    { toFun := fun x => r • p x
      map_zero' := by
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul, map_zero p,
          mul_zero]
      add_le_max' := fun x y => by
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul, ←
          mul_max_of_nonneg _ _ NNReal.zero_le_coe]
        /-
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : AddGroup E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : NonarchAddGroupSeminorm E
          x y : E
          ⊢ LE.le (HMul.hMul (↑(HSMul.hSMul r 1)) (p (HAdd.hAdd x y))) (HMul.hMul (↑(HSM …
        -/
        gcongr
        /-
          case h
          R : Type u_1
          R' : Type u_2
          E : Type u_3
          F : Type u_4
          G : Type u_5
          inst✝³ : AddGroup E
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : NonarchAddGroupSeminorm E
          x y : E
          ⊢ LE.le (p (HAdd.hAdd x y)) (Max.max (p x) (p y))
        -/
        apply map_add_le_max
        /-
          🎉 no goals
        -/
                          /-
                            R : Type u_1
                            R' : Type u_2
                            E : Type u_3
                            F : Type u_4
                            G : Type u_5
                            inst✝³ : AddGroup E
                            inst✝² : SMul R Real
                            inst✝¹ : SMul R NNReal
                            inst✝ : IsScalarTower R NNReal Real
                            r : R
                            p : NonarchAddGroupSeminorm E
                            x : E
                            ⊢ Eq ({ toFun := fun x => HSMul.hSMul r (p x), map_zero' := ⋯ }.toFun (Neg.neg …
                          -/
      neg' := fun x => by simp_rw [map_neg_eq_map p] }⟩
                          /-
                            🎉 no goals
                          -/


instance [SMul R' ℝ] [SMul R' ℝ≥0] [IsScalarTower R' ℝ≥0 ℝ] [SMul R R'] [IsScalarTower R R' ℝ] :
    IsScalarTower R R' (NonarchAddGroupSeminorm E) :=
  ⟨fun r a p => ext fun x => smul_assoc r a <| p x⟩


@[simp, norm_cast]
theorem coe_smul (r : R) (p : NonarchAddGroupSeminorm E) : ⇑(r • p) = r • ⇑p :=
  rfl


@[simp]
theorem smul_apply (r : R) (p : NonarchAddGroupSeminorm E) (x : E) : (r • p) x = r • p x :=
  rfl


theorem smul_sup (r : R) (p q : NonarchAddGroupSeminorm E) : r • (p ⊔ q) = r • p ⊔ r • q :=
  have Real.smul_max : ∀ x y : ℝ, r • max x y = max (r • x) (r • y) := fun x y => by
    simpa only [← smul_eq_mul, ← NNReal.smul_def, smul_one_smul ℝ≥0 r (_ : ℝ)] using
      mul_max_of_nonneg x y (r • (1 : ℝ≥0) : ℝ≥0).coe_nonneg
  ext fun _ => Real.smul_max _ _


@[to_additive]
instance funLike : FunLike (GroupNorm E) E ℝ where
  coe f := f.toFun
                             /-
                               R : Type u_1
                               R' : Type u_2
                               E : Type u_3
                               F : Type u_4
                               G : Type u_5
                               inst✝ : Group E
                               p q f g : GroupNorm E
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨_, _, _, _⟩, _⟩ := f; cases g; congr
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[to_additive]
instance groupNormClass : GroupNormClass (GroupNorm E) E ℝ where
  map_one_eq_zero f := f.map_one'
  map_mul_le_add f := f.mul_le'
  map_inv_eq_map f := f.inv'
  eq_one_of_map_eq_zero f := f.eq_one_of_map_eq_zero' _

-- Porting note: `simpNF` told me the left-hand side simplified to this

@[to_additive (attr := simp)]
theorem toGroupSeminorm_eq_coe : ⇑p.toGroupSeminorm = p :=
  rfl


@[to_additive]
instance : PartialOrder (GroupNorm E) :=
  PartialOrder.lift _ DFunLike.coe_injective


@[to_additive]
instance : Add (GroupNorm E) :=
  ⟨fun p q =>
    { p.toGroupSeminorm + q.toGroupSeminorm with
      eq_one_of_map_eq_zero' := fun _x hx =>
        of_not_not fun h => hx.not_gt <| add_pos (map_pos_of_ne_one p h) (map_pos_of_ne_one q h) }⟩


@[to_additive (attr := simp)]
theorem add_apply (x : E) : (p + q) x = p x + q x :=
  rfl

-- TODO: define `SupSet`

@[to_additive]
instance : Max (GroupNorm E) :=
  ⟨fun p q =>
    { p.toGroupSeminorm ⊔ q.toGroupSeminorm with
      eq_one_of_map_eq_zero' := fun _x hx =>
        of_not_not fun h => hx.not_gt <| lt_sup_iff.2 <| Or.inl <| map_pos_of_ne_one p h }⟩


@[to_additive]
instance : SemilatticeSup (GroupNorm E) :=
  DFunLike.coe_injective.semilatticeSup _ coe_sup


instance : One (AddGroupNorm E) :=
  ⟨{ (1 : AddGroupSeminorm E) with
      eq_zero_of_map_eq_zero' := fun _x => zero_ne_one.ite_eq_left_iff.1 }⟩


@[simp]
theorem apply_one (x : E) : (1 : AddGroupNorm E) x = if x = 0 then 0 else 1 :=
  rfl


instance : Inhabited (AddGroupNorm E) :=
  ⟨1⟩


instance _root_.AddGroupNorm.toOne [AddGroup E] [DecidableEq E] : One (AddGroupNorm E) :=
  ⟨{ (1 : AddGroupSeminorm E) with
    eq_zero_of_map_eq_zero' := fun _ => zero_ne_one.ite_eq_left_iff.1 }⟩


@[to_additive existing AddGroupNorm.toOne]
instance toOne : One (GroupNorm E) :=
  ⟨{ (1 : GroupSeminorm E) with eq_one_of_map_eq_zero' := fun _ => zero_ne_one.ite_eq_left_iff.1 }⟩


@[to_additive (attr := simp) existing AddGroupNorm.apply_one]
theorem apply_one (x : E) : (1 : GroupNorm E) x = if x = 1 then 0 else 1 :=
  rfl


@[to_additive existing]
instance : Inhabited (GroupNorm E) :=
  ⟨1⟩


instance funLike : FunLike (NonarchAddGroupNorm E) E ℝ where
  coe f := f.toFun
                             /-
                               R : Type u_1
                               R' : Type u_2
                               E : Type u_3
                               F : Type u_4
                               G : Type u_5
                               inst✝ : AddGroup E
                               p q f g : NonarchAddGroupNorm E
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨⟨_, _⟩, _, _⟩, _⟩ := f; cases g; congr
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance nonarchAddGroupNormClass : NonarchAddGroupNormClass (NonarchAddGroupNorm E) E where
  map_add_le_max f := f.add_le_max'
  map_zero f := f.map_zero'
  map_neg_eq_map' f := f.neg'
  eq_zero_of_map_eq_zero f := f.eq_zero_of_map_eq_zero' _

-- Porting note: `simpNF` told me the left-hand side simplified to this

@[simp]
theorem toNonarchAddGroupSeminorm_eq_coe : ⇑p.toNonarchAddGroupSeminorm = p :=
  rfl


noncomputable instance : PartialOrder (NonarchAddGroupNorm E) :=
  PartialOrder.lift _ DFunLike.coe_injective


instance : Max (NonarchAddGroupNorm E) :=
  ⟨fun p q =>
    { p.toNonarchAddGroupSeminorm ⊔ q.toNonarchAddGroupSeminorm with
      eq_zero_of_map_eq_zero' := fun _x hx =>
        of_not_not fun h => hx.not_gt <| lt_sup_iff.2 <| Or.inl <| map_pos_of_ne_zero p h }⟩


noncomputable instance : SemilatticeSup (NonarchAddGroupNorm E) :=
  DFunLike.coe_injective.semilatticeSup _ coe_sup


instance [DecidableEq E] : One (NonarchAddGroupNorm E) :=
  ⟨{ (1 : NonarchAddGroupSeminorm E) with
      eq_zero_of_map_eq_zero' := fun _ => zero_ne_one.ite_eq_left_iff.1 }⟩


@[simp]
theorem apply_one [DecidableEq E] (x : E) :
    (1 : NonarchAddGroupNorm E) x = if x = 0 then 0 else 1 :=
  rfl


instance [DecidableEq E] : Inhabited (NonarchAddGroupNorm E) :=
  ⟨1⟩


