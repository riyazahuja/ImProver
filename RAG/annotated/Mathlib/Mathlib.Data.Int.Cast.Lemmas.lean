/-- Coercion `ℕ → ℤ` as a `RingHom`. -/
def ofNatHom : ℕ →+* ℤ :=
  Nat.castRingHom ℤ


@[simp, norm_cast]
theorem cast_ite [AddGroupWithOne α] (P : Prop) [Decidable P] (m n : ℤ) :
    ((ite P m n : ℤ) : α) = ite P (m : α) (n : α) :=
  apply_ite _ _ _ _


/-- `coe : ℤ → α` as an `AddMonoidHom`. -/
def castAddHom (α : Type*) [AddGroupWithOne α] : ℤ →+ α where
  toFun := Int.cast
  map_zero' := cast_zero
  map_add' := cast_add


@[simp] lemma coe_castAddHom : ⇑(castAddHom α) = fun x : ℤ => (x : α) := rfl


lemma _root_.Even.intCast {n : ℤ} (h : Even n) : Even (n : α) := h.map (castAddHom α)


@[simp] lemma cast_eq_zero : (n : α) = 0 ↔ n = 0 where
  mp h := by
    /-
      α : Type u_3
      inst✝¹ : AddGroupWithOne α
      inst✝ : CharZero α
      n : Int
      h : Eq (↑n) 0
      ⊢ Eq n 0
    -/
    cases n
      /-
        case ofNat
        α : Type u_3
        inst✝¹ : AddGroupWithOne α
        inst✝ : CharZero α
        a✝ : Nat
        h : Eq (↑(Int.ofNat a✝)) 0
        ⊢ Eq (Int.ofNat a✝) 0
      -/
    · erw [Int.cast_natCast] at h
      /-
        case ofNat
        α : Type u_3
        inst✝¹ : AddGroupWithOne α
        inst✝ : CharZero α
        a✝ : Nat
        h : Eq (↑a✝) 0
        ⊢ Eq (Int.ofNat a✝) 0
      -/
      exact congr_arg _ (Nat.cast_eq_zero.1 h)
      /-
        🎉 no goals
      -/
      /-
        case negSucc
        α : Type u_3
        inst✝¹ : AddGroupWithOne α
        inst✝ : CharZero α
        a✝ : Nat
        h : Eq (↑(Int.negSucc a✝)) 0
        ⊢ Eq (Int.negSucc a✝) 0
      -/
    · rw [cast_negSucc, neg_eq_zero, Nat.cast_eq_zero] at h
      /-
        case negSucc
        α : Type u_3
        inst✝¹ : AddGroupWithOne α
        inst✝ : CharZero α
        a✝ : Nat
        h : Eq (HAdd.hAdd a✝ 1) 0
        ⊢ Eq (Int.negSucc a✝) 0
      -/
      contradiction
      /-
        🎉 no goals
      -/
              /-
                α : Type u_3
                inst✝¹ : AddGroupWithOne α
                inst✝ : CharZero α
                n : Int
                h : Eq n 0
                ⊢ Eq (↑n) 0
              -/
  mpr h := by rw [h, cast_zero]
              /-
                🎉 no goals
              -/


@[simp, norm_cast]
                                           /-
                                             α : Type u_3
                                             inst✝¹ : AddGroupWithOne α
                                             inst✝ : CharZero α
                                             m n : Int
                                             ⊢ Iff (Eq ↑m ↑n) (Eq m n)
                                           -/
lemma cast_inj : (m : α) = n ↔ m = n := by rw [← sub_eq_zero, ← cast_sub, cast_eq_zero, sub_eq_zero]
                                           /-
                                             🎉 no goals
                                           -/


lemma cast_injective : Injective (Int.cast : ℤ → α) := fun _ _ ↦ cast_inj.1


lemma cast_ne_zero : (n : α) ≠ 0 ↔ n ≠ 0 := not_congr cast_eq_zero


                                                      /-
                                                        α : Type u_3
                                                        inst✝¹ : AddGroupWithOne α
                                                        inst✝ : CharZero α
                                                        n : Int
                                                        ⊢ Iff (Eq (↑n) 1) (Eq n 1)
                                                      -/
@[simp] lemma cast_eq_one : (n : α) = 1 ↔ n = 1 := by rw [← cast_one, cast_inj]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma cast_ne_one : (n : α) ≠ 1 ↔ n ≠ 1 := cast_eq_one.not


variable (α) in
/-- `coe : ℤ → α` as a `RingHom`. -/
def castRingHom : ℤ →+* α where
  toFun := Int.cast
  map_zero' := cast_zero
  map_add' := cast_add
  map_one' := cast_one
  map_mul' := cast_mul


@[simp] lemma coe_castRingHom : ⇑(castRingHom α) = fun x : ℤ ↦ (x : α) := rfl


lemma cast_commute : ∀ (n : ℤ) (a : α), Commute ↑n a
                     /-
                       α : Type u_3
                       inst✝ : NonAssocRing α
                       n : Nat
                       x : α
                       ⊢ Commute (↑↑n) x
                     -/
  | (n : ℕ), x => by simpa using n.cast_commute x
                     /-
                       🎉 no goals
                     -/
  | -[n+1], x => by
    simpa only [cast_negSucc, Commute.neg_left_iff, Commute.neg_right_iff] using
      (n + 1).cast_commute (-x)


lemma cast_comm (n : ℤ) (x : α) : n * x = x * n := (cast_commute ..).eq


lemma commute_cast (a : α) (n : ℤ) : Commute a n := (cast_commute ..).symm


@[simp] lemma _root_.zsmul_eq_mul (a : α) : ∀ n : ℤ, n • a = n * a
                  /-
                    α : Type u_3
                    inst✝ : NonAssocRing α
                    a : α
                    n : Nat
                    ⊢ Eq (HSMul.hSMul (↑n) a) (HMul.hMul (↑↑n) a)
                  -/
  | (n : ℕ) => by rw [natCast_zsmul, nsmul_eq_mul, Int.cast_natCast]
                  /-
                    🎉 no goals
                  -/
                 /-
                   α : Type u_3
                   inst✝ : NonAssocRing α
                   a : α
                   n : Nat
                   ⊢ Eq (HSMul.hSMul (Int.negSucc n) a) (HMul.hMul (↑(Int.negSucc n)) a)
                 -/
  | -[n+1] => by simp [Nat.cast_succ, neg_add_rev, Int.cast_negSucc, add_mul]
                 /-
                   🎉 no goals
                 -/


lemma _root_.zsmul_eq_mul' (a : α) (n : ℤ) : n • a = a * n := by
  /-
    α : Type u_3
    inst✝ : NonAssocRing α
    a : α
    n : Int
    ⊢ Eq (HSMul.hSMul n a) (HMul.hMul a ↑n)
  -/
  rw [zsmul_eq_mul, (n.cast_commute a).eq]
  /-
    🎉 no goals
  -/


lemma _root_.Odd.intCast (hn : Odd n) : Odd (n : α) := hn.map (castRingHom α)


theorem cast_dvd_cast [CommRing α] (m n : ℤ) (h : m ∣ n) : (m : α) ∣ (n : α) :=
  RingHom.map_dvd (Int.castRingHom α) h


@[deprecated (since := "2024-05-25")] alias coe_int_dvd := cast_dvd_cast


@[simp] lemma intCast_mul_right (h : SemiconjBy a x y) (n : ℤ) : SemiconjBy a (n * x) (n * y) :=
  SemiconjBy.mul_right (Int.commute_cast _ _) h


@[simp] lemma intCast_mul_left (h : SemiconjBy a x y) (n : ℤ) : SemiconjBy (n * a) x y :=
  SemiconjBy.mul_left (Int.cast_commute _ _) h


@[simp] lemma intCast_mul_intCast_mul (h : SemiconjBy a x y) (m n : ℤ) :
    SemiconjBy (m * a) (n * x) (n * y) := (h.intCast_mul_left m).intCast_mul_right n


@[deprecated (since := "2024-05-27")] alias cast_int_mul_right := intCast_mul_right

@[deprecated (since := "2024-05-27")] alias cast_int_mul_left := intCast_mul_left

@[deprecated (since := "2024-05-27")] alias cast_int_mul_cast_int_mul := intCast_mul_intCast_mul


@[simp] lemma intCast_left : Commute (n : α) a := Int.cast_commute _ _


@[simp] lemma intCast_right : Commute a n := Int.commute_cast _ _


@[deprecated (since := "2024-05-27")] alias cast_int_right := intCast_right

@[deprecated (since := "2024-05-27")] alias cast_int_left := intCast_left


@[simp] lemma intCast_mul_right (h : Commute a b) (m : ℤ) : Commute a (m * b) :=
  SemiconjBy.intCast_mul_right h m


@[simp] lemma intCast_mul_left (h : Commute a b) (m : ℤ) : Commute (m  * a) b :=
  SemiconjBy.intCast_mul_left h m


lemma intCast_mul_intCast_mul (h : Commute a b) (m n : ℤ) : Commute (m * a) (n * b) :=
  SemiconjBy.intCast_mul_intCast_mul h m n


lemma self_intCast_mul : Commute a (n * a : α) := (Commute.refl a).intCast_mul_right n


lemma intCast_mul_self : Commute ((n : α) * a) a := (Commute.refl a).intCast_mul_left n


lemma self_intCast_mul_intCast_mul : Commute (m * a : α) (n * a : α) :=
  (Commute.refl a).intCast_mul_intCast_mul m n


@[deprecated (since := "2024-05-27")] alias cast_int_mul_cast_int_mul := intCast_mul_intCast_mul

@[deprecated (since := "2024-05-27")] alias self_cast_int_mul := self_intCast_mul

@[deprecated (since := "2024-05-27")] alias cast_int_mul_self := intCast_mul_self

@[deprecated (since := "2024-05-27")]
alias self_cast_int_mul_cast_int_mul := self_intCast_mul_intCast_mul


/-- Two additive monoid homomorphisms `f`, `g` from `ℤ` to an additive monoid are equal
if `f 1 = g 1`. -/
@[ext high]
theorem ext_int [AddMonoid A] {f g : ℤ →+ A} (h1 : f 1 = g 1) : f = g :=
  have : f.comp (Int.ofNatHom : ℕ →+ ℤ) = g.comp (Int.ofNatHom : ℕ →+ ℤ) := ext_nat' _ _ h1
  have this' : ∀ n : ℕ, f n = g n := DFunLike.ext_iff.1 this
  ext fun n => match n with
  | (n : ℕ) => this' n
  | .negSucc n => eq_on_neg _ _ (this' <| n + 1)


theorem eq_intCastAddHom (f : ℤ →+ A) (h1 : f 1 = 1) : f = Int.castAddHom A :=
                /-
                  A : Type u_5
                  inst✝ : AddGroupWithOne A
                  f : AddMonoidHom Int A
                  h1 : Eq (f 1) 1
                  ⊢ Eq (f 1) ((Int.castAddHom A) 1)
                -/
  ext_int <| by simp [h1]
                /-
                  🎉 no goals
                -/


@[deprecated (since := "2024-04-17")]
alias eq_int_castAddHom := eq_intCastAddHom


theorem eq_intCast' [AddGroupWithOne α] [FunLike F ℤ α] [AddMonoidHomClass F ℤ α]
    (f : F) (h₁ : f 1 = 1) :
    ∀ n : ℤ, f n = n :=
  DFunLike.ext_iff.1 <| (f : ℤ →+ α).eq_intCastAddHom h₁


/-- This version is primed so that the `RingHomClass` versions aren't. -/
theorem map_intCast' [AddGroupWithOne α] [AddGroupWithOne β] [FunLike F α β]
    [AddMonoidHomClass F α β] (f : F) (h₁ : f 1 = 1) : ∀ n : ℤ, f n = n :=
                                                          /-
                                                            F : Type u_1
                                                            α : Type u_3
                                                            β : Type u_4
                                                            inst✝³ : AddGroupWithOne α
                                                            inst✝² : AddGroupWithOne β
                                                            inst✝¹ : FunLike F α β
                                                            inst✝ : AddMonoidHomClass F α β
                                                            f : F
                                                            h₁ : Eq (f 1) 1
                                                            ⊢ Eq (((↑f).comp (Int.castAddHom α)) 1) 1
                                                          -/
  eq_intCast' ((f : α →+ β).comp <| Int.castAddHom _) (by simpa)
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem Int.castAddHom_int : Int.castAddHom ℤ = AddMonoidHom.id ℤ :=
  ((AddMonoidHom.id ℤ).eq_intCastAddHom rfl).symm


@[ext]
theorem ext_mint {f g : Multiplicative ℤ →* M} (h1 : f (ofAdd 1) = g (ofAdd 1)) : f = g :=
  MonoidHom.toAdditive''.injective <| AddMonoidHom.ext_int <| Additive.toMul.injective h1


/-- If two `MonoidHom`s agree on `-1` and the naturals then they are equal. -/
@[ext]
theorem ext_int {f g : ℤ →* M} (h_neg_one : f (-1) = g (-1))
    (h_nat : f.comp Int.ofNatHom.toMonoidHom = g.comp Int.ofNatHom.toMonoidHom) : f = g := by
  /-
    M : Type u_5
    inst✝ : Monoid M
    f g : MonoidHom Int M
    h_neg_one : Eq (f (-1)) (g (-1))
    h_nat : Eq (f.comp ↑Int.ofNatHom) (g.comp ↑Int.ofNatHom)
    ⊢ Eq f g
  -/
  ext (x | x)
    /-
      case h.ofNat
      M : Type u_5
      inst✝ : Monoid M
      f g : MonoidHom Int M
      h_neg_one : Eq (f (-1)) (g (-1))
      h_nat : Eq (f.comp ↑Int.ofNatHom) (g.comp ↑Int.ofNatHom)
      x : Nat
      ⊢ Eq (f (Int.ofNat x)) (g (Int.ofNat x))
    -/
  · exact (DFunLike.congr_fun h_nat x : _)
    /-
      🎉 no goals
    -/
    /-
      case h.negSucc
      M : Type u_5
      inst✝ : Monoid M
      f g : MonoidHom Int M
      h_neg_one : Eq (f (-1)) (g (-1))
      h_nat : Eq (f.comp ↑Int.ofNatHom) (g.comp ↑Int.ofNatHom)
      x : Nat
      ⊢ Eq (f (Int.negSucc x)) (g (Int.negSucc x))
    -/
  · rw [Int.negSucc_eq, ← neg_one_mul, f.map_mul, g.map_mul]
    /-
      case h.negSucc
      M : Type u_5
      inst✝ : Monoid M
      f g : MonoidHom Int M
      h_neg_one : Eq (f (-1)) (g (-1))
      h_nat : Eq (f.comp ↑Int.ofNatHom) (g.comp ↑Int.ofNatHom)
      x : Nat
      ⊢ Eq (HMul.hMul (f (-1)) (f (HAdd.hAdd (↑x) 1))) (HMul.hMul (g (-1)) (g (HAdd. …
    -/
    congr 1
    /-
      case h.negSucc.e_a
      M : Type u_5
      inst✝ : Monoid M
      f g : MonoidHom Int M
      h_neg_one : Eq (f (-1)) (g (-1))
      h_nat : Eq (f.comp ↑Int.ofNatHom) (g.comp ↑Int.ofNatHom)
      x : Nat
      ⊢ Eq (f (HAdd.hAdd (↑x) 1)) (g (HAdd.hAdd (↑x) 1))
    -/
    exact mod_cast (DFunLike.congr_fun h_nat (x + 1) : _)
    /-
      🎉 no goals
    -/


/-- If two `MonoidWithZeroHom`s agree on `-1` and the naturals then they are equal. -/
@[ext]
theorem ext_int {f g : ℤ →*₀ M} (h_neg_one : f (-1) = g (-1))
    (h_nat : f.comp Int.ofNatHom.toMonoidWithZeroHom = g.comp Int.ofNatHom.toMonoidWithZeroHom) :
    f = g :=
  toMonoidHom_injective <| MonoidHom.ext_int h_neg_one <|
    MonoidHom.ext (DFunLike.congr_fun h_nat : _)


/-- If two `MonoidWithZeroHom`s agree on `-1` and the _positive_ naturals then they are equal. -/
theorem ext_int' [MonoidWithZero α] [FunLike F ℤ α] [MonoidWithZeroHomClass F ℤ α] {f g : F}
    (h_neg_one : f (-1) = g (-1)) (h_pos : ∀ n : ℕ, 0 < n → f n = g n) : f = g :=
  (DFunLike.ext _ _) fun n =>
    haveI :=
      DFunLike.congr_fun
        (@MonoidWithZeroHom.ext_int _ _ (f : ℤ →*₀ α) (g : ℤ →*₀ α) h_neg_one <|
          MonoidWithZeroHom.ext_nat (h_pos _))
        n
    this


/-- Additive homomorphisms from `ℤ` are defined by the image of `1`. -/
def zmultiplesHom : β ≃ (ℤ →+ β) where
  toFun x :=
  { toFun := fun n => n • x
    map_zero' := zero_zsmul x
    map_add' := fun _ _ => add_zsmul _ _ _ }
  invFun f := f 1
  left_inv := one_zsmul
  right_inv f := AddMonoidHom.ext_int <| one_zsmul (f 1)


/-- Monoid homomorphisms from `Multiplicative ℤ` are defined by the image
of `Multiplicative.ofAdd 1`. -/
@[to_additive existing]
def zpowersHom : α ≃ (Multiplicative ℤ →* α) :=
  ofMul.trans <| (zmultiplesHom _).trans <| AddMonoidHom.toMultiplicative''


lemma zmultiplesHom_apply (x : β) (n : ℤ) : zmultiplesHom β x n = n • x := rfl


lemma zmultiplesHom_symm_apply (f : ℤ →+ β) : (zmultiplesHom β).symm f = f 1 := rfl


@[to_additive existing (attr := simp)]
lemma zpowersHom_apply (x : α) (n : Multiplicative ℤ) : zpowersHom α x n = x ^ n.toAdd := rfl


@[to_additive existing (attr := simp)]
lemma zpowersHom_symm_apply (f : Multiplicative ℤ →* α) :
    (zpowersHom α).symm f = f (ofAdd 1) := rfl


lemma MonoidHom.apply_mint (f : Multiplicative ℤ →* α) (n : Multiplicative ℤ) :
    f n = f (ofAdd 1) ^ n.toAdd := by
  /-
    α : Type u_3
    inst✝ : Group α
    f : MonoidHom (Multiplicative Int) α
    n : Multiplicative Int
    ⊢ Eq (f n) (HPow.hPow (f (Multiplicative.ofAdd 1)) (Multiplicative.toAdd n))
  -/
  rw [← zpowersHom_symm_apply, ← zpowersHom_apply, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma AddMonoidHom.apply_int (f : ℤ →+ β) (n : ℤ) : f n = n • f 1 := by
  /-
    β : Type u_4
    inst✝ : AddGroup β
    f : AddMonoidHom Int β
    n : Int
    ⊢ Eq (f n) (HSMul.hSMul n (f 1))
  -/
  rw [← zmultiplesHom_symm_apply, ← zmultiplesHom_apply, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- If `α` is commutative, `zmultiplesHom` is an additive equivalence. -/
def zmultiplesAddHom : β ≃+ (ℤ →+ β) :=
                                                                             /-
                                                                               F : Type u_1
                                                                               ι : Type u_2
                                                                               α : Type u_3
                                                                               β : Type u_4
                                                                               inst✝¹ : CommGroup α
                                                                               inst✝ : AddCommGroup β
                                                                               a b : β
                                                                               n : Int
                                                                               ⊢ Eq ((__src✝.toFun (HAdd.hAdd a b)) n) ((HAdd.hAdd (__src✝.toFun a) (__src✝.t …
                                                                             -/
  { zmultiplesHom β with map_add' := fun a b => AddMonoidHom.ext fun n => by simp [zsmul_add] }
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- If `α` is commutative, `zpowersHom` is a multiplicative equivalence. -/
def zpowersMulHom : α ≃* (Multiplicative ℤ →* α) :=
                                                                       /-
                                                                         F : Type u_1
                                                                         ι : Type u_2
                                                                         α : Type u_3
                                                                         β : Type u_4
                                                                         inst✝¹ : CommGroup α
                                                                         inst✝ : AddCommGroup β
                                                                         a b : α
                                                                         n : Multiplicative Int
                                                                         ⊢ Eq ((__src✝.toFun (HMul.hMul a b)) n) ((HMul.hMul (__src✝.toFun a) (__src✝.t …
                                                                       -/
  { zpowersHom α with map_mul' := fun a b => MonoidHom.ext fun n => by simp [mul_zpow] }
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
lemma zpowersMulHom_apply (x : α) (n : Multiplicative ℤ) : zpowersMulHom α x n = x ^ n.toAdd := rfl


@[simp]
lemma zpowersMulHom_symm_apply (f : Multiplicative ℤ →* α) :
    (zpowersMulHom α).symm f = f (ofAdd 1) := rfl


@[simp] lemma zmultiplesAddHom_apply (x : β) (n : ℤ) : zmultiplesAddHom β x n = n • x := rfl


@[simp] lemma zmultiplesAddHom_symm_apply (f : ℤ →+ β) : (zmultiplesAddHom β).symm f = f 1 := rfl


@[simp]
theorem eq_intCast [FunLike F ℤ α] [RingHomClass F ℤ α] (f : F) (n : ℤ) : f n = n :=
  eq_intCast' f (map_one _) n


@[simp]
theorem map_intCast [FunLike F α β] [RingHomClass F α β] (f : F) (n : ℤ) : f n = n :=
  eq_intCast ((f : α →+* β).comp (Int.castRingHom α)) n


theorem eq_intCast' (f : ℤ →+* α) : f = Int.castRingHom α :=
  RingHom.ext <| eq_intCast f


theorem ext_int {R : Type*} [NonAssocSemiring R] (f g : ℤ →+* R) : f = g :=
  coe_addMonoidHom_injective <| AddMonoidHom.ext_int <| f.map_one.trans g.map_one.symm


instance Int.subsingleton_ringHom {R : Type*} [NonAssocSemiring R] : Subsingleton (ℤ →+* R) :=
  ⟨RingHom.ext_int⟩


@[simp]
theorem Int.castRingHom_int : Int.castRingHom ℤ = RingHom.id ℤ :=
  (RingHom.id ℤ).eq_intCast'.symm

