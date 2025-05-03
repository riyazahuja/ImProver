/-- The Pontryagin dual of `A` is the group of continuous homomorphism `A → Circle`. -/
def PontryaginDual :=
  ContinuousMonoidHom A Circle

-- Porting note: `deriving` doesn't derive these instances

instance : TopologicalSpace (PontryaginDual A) :=
  (inferInstance : TopologicalSpace (ContinuousMonoidHom A Circle))


instance : T2Space (PontryaginDual A) :=
  (inferInstance : T2Space (ContinuousMonoidHom A Circle))

-- Porting note: instance is now noncomputable

noncomputable instance : CommGroup (PontryaginDual A) :=
  (inferInstance : CommGroup (ContinuousMonoidHom A Circle))


instance : TopologicalGroup (PontryaginDual A) :=
  (inferInstance : TopologicalGroup (ContinuousMonoidHom A Circle))

-- Porting note: instance is now noncomputable

noncomputable instance : Inhabited (PontryaginDual A) :=
  (inferInstance : Inhabited (ContinuousMonoidHom A Circle))


instance [LocallyCompactSpace H] : LocallyCompactSpace (PontryaginDual H) := by
  let Vn : ℕ → Set Circle :=
    fun n ↦ Circle.exp '' { x | |x| < Real.pi / 2 ^ (n + 1)}
  have hVn : ∀ n x, x ∈ Vn n ↔ |Complex.arg x| < Real.pi / 2 ^ (n + 1) := by
    refine fun n x ↦ ⟨?_, fun hx ↦ ⟨Complex.arg x, hx, Circle.exp_arg x⟩⟩
    rintro ⟨t, ht : |t| < _, rfl⟩
    have ht' := ht.trans_le (div_le_self Real.pi_nonneg (one_le_pow₀ one_le_two))
    rwa [Circle.arg_exp (neg_lt_of_abs_lt ht') (lt_of_abs_lt ht').le]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝¹² : Monoid A
    inst✝¹¹ : Monoid B
    inst✝¹⁰ : Monoid C
    inst✝⁹ : CommGroup G
    inst✝⁸ : Group H
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace C
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalGroup G
    inst✝¹ : TopologicalGroup H
    inst✝ : LocallyCompactSpace H
    Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
    hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
    ⊢ LocallyCompactSpace (PontryaginDual H)
  -/
  refine ContinuousMonoidHom.locallyCompactSpace_of_hasBasis Vn ?_ ?_
    /-
      case refine_1
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      ⊢ ∀ {n : Nat} {x : Circle}, Membership.mem (Vn n) x → Membership.mem (Vn n) (H …
    -/
  · intro n x h1 h2
    /-
      case refine_1
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      n : Nat
      x : Circle
      h1 : Membership.mem (Vn n) x
      h2 : Membership.mem (Vn n) (HMul.hMul x x)
      ⊢ Membership.mem (Vn (HAdd.hAdd n 1)) x
    -/
    rw [hVn] at h1 h2 ⊢
    rwa [Circle.coe_mul, Complex.arg_mul x.coe_ne_zero x.coe_ne_zero,
      ← two_mul, abs_mul, abs_two, ← lt_div_iff₀' two_pos, div_div, ← pow_succ] at h2
    /-
      case refine_1
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      n : Nat
      x : Circle
      h1 : LT.lt (abs (↑x).arg) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd n 1)))
      h2 : LT.lt (abs (HMul.hMul ↑x ↑x).arg) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.h …
      ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (↑x).arg (↑x). …
    -/
    apply Set.Ioo_subset_Ioc_self
    /-
      case refine_1.a
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      n : Nat
      x : Circle
      h1 : LT.lt (abs (↑x).arg) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd n 1)))
      h2 : LT.lt (abs (HMul.hMul ↑x ↑x).arg) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.h …
      ⊢ Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) (HAdd.hAdd (↑x).arg (↑x). …
    -/
    rw [← two_mul, Set.mem_Ioo, ← abs_lt, abs_mul, abs_two, ← lt_div_iff₀' two_pos]
    exact h1.trans_le
      (div_le_div_of_nonneg_left Real.pi_nonneg two_pos (le_self_pow₀ one_le_two n.succ_ne_zero))
    /-
      case refine_2
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      ⊢ (nhds 1).HasBasis (fun x => True) Vn
    -/
  · rw [← Circle.exp_zero, ← isLocalHomeomorph_circleExp.map_nhds_eq 0]
    refine ((nhds_basis_zero_abs_sub_lt ℝ).to_hasBasis
        (fun x hx ↦ ⟨Nat.ceil (Real.pi / x), trivial, fun t ht ↦ ?_⟩)
          fun k _ ↦ ⟨Real.pi / 2 ^ (k + 1), by positivity, le_rfl⟩).map Circle.exp
    /-
      case refine_2
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      x : Real
      hx : LT.lt 0 x
      t : Real
      ht : Membership.mem (setOf fun b => LT.lt (abs b) (HDiv.hDiv Real.pi (HPow.hPo …
      ⊢ Membership.mem (setOf fun b => LT.lt (abs b) x) t
    -/
    rw [Set.mem_setOf_eq] at ht ⊢
    /-
      case refine_2
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      x : Real
      hx : LT.lt 0 x
      t : Real
      ht : LT.lt (abs t) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd (Nat.ceil (HDiv. …
      ⊢ LT.lt (abs t) x
    -/
    refine lt_of_lt_of_le ht ?_
    /-
      case refine_2
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      x : Real
      hx : LT.lt 0 x
      t : Real
      ht : LT.lt (abs t) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd (Nat.ceil (HDiv. …
      ⊢ LE.le (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd (Nat.ceil (HDiv.hDiv Real.p …
    -/
    rw [div_le_iff₀' (pow_pos two_pos _), ← div_le_iff₀ hx]
    /-
      case refine_2
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      x : Real
      hx : LT.lt 0 x
      t : Real
      ht : LT.lt (abs t) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd (Nat.ceil (HDiv. …
      ⊢ LE.le (HDiv.hDiv Real.pi x) (HPow.hPow 2 (HAdd.hAdd (Nat.ceil (HDiv.hDiv Rea …
    -/
    refine (Nat.le_ceil (Real.pi / x)).trans ?_
    /-
      case refine_2
      A : Type u_1
      B : Type u_2
      C : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹² : Monoid A
      inst✝¹¹ : Monoid B
      inst✝¹⁰ : Monoid C
      inst✝⁹ : CommGroup G
      inst✝⁸ : Group H
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace C
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalSpace H
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalGroup H
      inst✝ : LocallyCompactSpace H
      Vn : Nat → Set Circle := fun n => Set.image (⇑Circle.exp) (setOf fun x => LT.l …
      hVn : ∀ (n : Nat) (x : Circle), Iff (Membership.mem (Vn n) x) (LT.lt (abs (↑x) …
      x : Real
      hx : LT.lt 0 x
      t : Real
      ht : LT.lt (abs t) (HDiv.hDiv Real.pi (HPow.hPow 2 (HAdd.hAdd (Nat.ceil (HDiv. …
      ⊢ LE.le (↑(Nat.ceil (HDiv.hDiv Real.pi x))) (HPow.hPow 2 (HAdd.hAdd (Nat.ceil  …
    -/
    exact_mod_cast (Nat.le_succ _).trans Nat.lt_two_pow_self.le
    /-
      🎉 no goals
    -/


instance : FunLike (PontryaginDual A) A Circle :=
  ContinuousMonoidHom.instFunLike


noncomputable instance instContinuousMapClass : ContinuousMapClass (PontryaginDual A) A Circle :=
  ContinuousMonoidHom.instContinuousMapClass


noncomputable instance instMonoidHomClass : MonoidHomClass (PontryaginDual A) A Circle :=
  ContinuousMonoidHom.instMonoidHomClass


/-- `PontryaginDual` is a contravariant functor. -/
noncomputable def map (f : ContinuousMonoidHom A B) :
    ContinuousMonoidHom (PontryaginDual B) (PontryaginDual A) :=
  f.compLeft Circle


@[simp]
theorem map_apply (f : ContinuousMonoidHom A B) (x : PontryaginDual B) (y : A) :
    map f x y = x (f y) :=
  rfl


@[simp]
theorem map_one : map (one A B) = one (PontryaginDual B) (PontryaginDual A) :=
  ext fun x => ext (fun _y => OneHomClass.map_one x)


@[simp]
theorem map_comp (g : ContinuousMonoidHom B C) (f : ContinuousMonoidHom A B) :
    map (comp g f) = ContinuousMonoidHom.comp (map f) (map g) :=
  ext fun _x => ext fun _y => rfl


@[simp]
nonrec theorem map_mul (f g : ContinuousMonoidHom A G) : map (f * g) = map f * map g :=
  ext fun x => ext fun y => map_mul x (f y) (g y)


/-- `ContinuousMonoidHom.dual` as a `ContinuousMonoidHom`. -/
noncomputable def mapHom [LocallyCompactSpace G] :
    ContinuousMonoidHom (ContinuousMonoidHom A G)
      (ContinuousMonoidHom (PontryaginDual G) (PontryaginDual A)) where
  toFun := map
  map_one' := map_one
  map_mul' := map_mul
  continuous_toFun := continuous_of_continuous_uncurry _ continuous_comp


