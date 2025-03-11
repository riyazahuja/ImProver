/-- A bundled map `f : G → H` such that `f (x + a) = f x + b` for all `x`.

One can think about `f` as a lift to `G` of a map between two `AddCircle`s. -/
structure AddConstMap (G H : Type*) [Add G] [Add H] (a : G) (b : H) where
  /-- The underlying function of an `AddConstMap`.
  Use automatic coercion to function instead. -/
  protected toFun : G → H
  /-- An `AddConstMap` satisfies `f (x + a) = f x + b`. Use `map_add_const` instead. -/
  map_add_const' (x : G) : toFun (x + a) = toFun x + b


@[inherit_doc]
scoped [AddConstMap] notation:25 G " →+c[" a ", " b "] " H => AddConstMap G H a b


/-- Typeclass for maps satisfying `f (x + a) = f x + b`.

Note that `a` and `b` are `outParam`s,
so one should not add instances like
`[AddConstMapClass F G H a b] : AddConstMapClass F G H (-a) (-b)`. -/
class AddConstMapClass (F : Type*) (G H : outParam Type*) [Add G] [Add H]
    (a : outParam G) (b : outParam H) [FunLike F G H] : Prop where
  /-- A map of `AddConstMapClass` class semiconjugates shift by `a` to the shift by `b`:
  `∀ x, f (x + a) = f x + b`. -/
  map_add_const (f : F) (x : G) : f (x + a) = f x + b


protected theorem semiconj [Add G] [Add H] [AddConstMapClass F G H a b] (f : F) :
    Semiconj f (· + a) (· + b) :=
  map_add_const f


@[scoped simp]
theorem map_add_nsmul [AddMonoid G] [AddMonoid H] [AddConstMapClass F G H a b]
    (f : F) (x : G) (n : ℕ) : f (x + n • a) = f x + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddMonoid G
    inst✝¹ : AddMonoid H
    inst✝ : AddConstMapClass F G H a b
    f : F
    x : G
    n : Nat
    ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul n a))) (HAdd.hAdd (f x) (HSMul.hSMul n b))
  -/
  simpa using (AddConstMapClass.semiconj f).iterate_right n x
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_add_nat' [AddMonoidWithOne G] [AddMonoid H] [AddConstMapClass F G H 1 b]
                                                            /-
                                                              F : Type u_1
                                                              G : Type u_2
                                                              H : Type u_3
                                                              inst✝³ : FunLike F G H
                                                              b : H
                                                              inst✝² : AddMonoidWithOne G
                                                              inst✝¹ : AddMonoid H
                                                              inst✝ : AddConstMapClass F G H 1 b
                                                              f : F
                                                              x : G
                                                              n : Nat
                                                              ⊢ Eq (f (HAdd.hAdd x ↑n)) (HAdd.hAdd (f x) (HSMul.hSMul n b))
                                                            -/
    (f : F) (x : G) (n : ℕ) : f (x + n) = f x + n • b := by simp [← map_add_nsmul]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem map_add_one [AddMonoidWithOne G] [Add H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) : f (x + 1) = f x + b := map_add_const f x


@[scoped simp]
theorem map_add_ofNat' [AddMonoidWithOne G] [AddMonoid H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) (n : ℕ) [n.AtLeastTwo] :
    f (x + no_index (OfNat.ofNat n)) = f x + (OfNat.ofNat n : ℕ) • b :=
  map_add_nat' f x n


theorem map_add_nat [AddMonoidWithOne G] [AddMonoidWithOne H] [AddConstMapClass F G H 1 1]
                                                        /-
                                                          F : Type u_1
                                                          G : Type u_2
                                                          H : Type u_3
                                                          inst✝³ : FunLike F G H
                                                          inst✝² : AddMonoidWithOne G
                                                          inst✝¹ : AddMonoidWithOne H
                                                          inst✝ : AddConstMapClass F G H 1 1
                                                          f : F
                                                          x : G
                                                          n : Nat
                                                          ⊢ Eq (f (HAdd.hAdd x ↑n)) (HAdd.hAdd (f x) ↑n)
                                                        -/
    (f : F) (x : G) (n : ℕ) : f (x + n) = f x + n := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem map_add_ofNat [AddMonoidWithOne G] [AddMonoidWithOne H] [AddConstMapClass F G H 1 1]
    (f : F) (x : G) (n : ℕ) [n.AtLeastTwo] :
    f (x + OfNat.ofNat n) = f x + OfNat.ofNat n := map_add_nat f x n


@[scoped simp]
theorem map_const [AddZeroClass G] [Add H] [AddConstMapClass F G H a b] (f : F) :
    f a = f 0 + b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddZeroClass G
    inst✝¹ : Add H
    inst✝ : AddConstMapClass F G H a b
    f : F
    ⊢ Eq (f a) (HAdd.hAdd (f 0) b)
  -/
  simpa using map_add_const f 0
  /-
    🎉 no goals
  -/


theorem map_one [AddZeroClass G] [One G] [Add H] [AddConstMapClass F G H 1 b] (f : F) :
    f 1 = f 0 + b :=
  map_const f


@[scoped simp]
theorem map_nsmul_const [AddMonoid G] [AddMonoid H] [AddConstMapClass F G H a b]
    (f : F) (n : ℕ) : f (n • a) = f 0 + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddMonoid G
    inst✝¹ : AddMonoid H
    inst✝ : AddConstMapClass F G H a b
    f : F
    n : Nat
    ⊢ Eq (f (HSMul.hSMul n a)) (HAdd.hAdd (f 0) (HSMul.hSMul n b))
  -/
  simpa using map_add_nsmul f 0 n
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_nat' [AddMonoidWithOne G] [AddMonoid H] [AddConstMapClass F G H 1 b]
    (f : F) (n : ℕ) : f n = f 0 + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    b : H
    inst✝² : AddMonoidWithOne G
    inst✝¹ : AddMonoid H
    inst✝ : AddConstMapClass F G H 1 b
    f : F
    n : Nat
    ⊢ Eq (f ↑n) (HAdd.hAdd (f 0) (HSMul.hSMul n b))
  -/
  simpa using map_add_nat' f 0 n
  /-
    🎉 no goals
  -/


theorem map_ofNat' [AddMonoidWithOne G] [AddMonoid H] [AddConstMapClass F G H 1 b]
    (f : F) (n : ℕ) [n.AtLeastTwo] :
    f (OfNat.ofNat n) = f 0 + (OfNat.ofNat n : ℕ) • b :=
  map_nat' f n


theorem map_nat [AddMonoidWithOne G] [AddMonoidWithOne H] [AddConstMapClass F G H 1 1]
                                          /-
                                            F : Type u_1
                                            G : Type u_2
                                            H : Type u_3
                                            inst✝³ : FunLike F G H
                                            inst✝² : AddMonoidWithOne G
                                            inst✝¹ : AddMonoidWithOne H
                                            inst✝ : AddConstMapClass F G H 1 1
                                            f : F
                                            n : Nat
                                            ⊢ Eq (f ↑n) (HAdd.hAdd (f 0) ↑n)
                                          -/
    (f : F) (n : ℕ) : f n = f 0 + n := by simp
                                          /-
                                            🎉 no goals
                                          -/


theorem map_ofNat [AddMonoidWithOne G] [AddMonoidWithOne H] [AddConstMapClass F G H 1 1]
    (f : F) (n : ℕ) [n.AtLeastTwo] :
    f (OfNat.ofNat n) = f 0 + OfNat.ofNat n := map_nat f n


@[scoped simp]
theorem map_const_add [AddCommSemigroup G] [Add H] [AddConstMapClass F G H a b]
    (f : F) (x : G) : f (a + x) = f x + b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddCommSemigroup G
    inst✝¹ : Add H
    inst✝ : AddConstMapClass F G H a b
    f : F
    x : G
    ⊢ Eq (f (HAdd.hAdd a x)) (HAdd.hAdd (f x) b)
  -/
  rw [add_comm, map_add_const]
  /-
    🎉 no goals
  -/


theorem map_one_add [AddCommMonoidWithOne G] [Add H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) : f (1 + x) = f x + b := map_const_add f x


@[scoped simp]
theorem map_nsmul_add [AddCommMonoid G] [AddMonoid H] [AddConstMapClass F G H a b]
    (f : F) (n : ℕ) (x : G) : f (n • a + x) = f x + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddCommMonoid G
    inst✝¹ : AddMonoid H
    inst✝ : AddConstMapClass F G H a b
    f : F
    n : Nat
    x : G
    ⊢ Eq (f (HAdd.hAdd (HSMul.hSMul n a) x)) (HAdd.hAdd (f x) (HSMul.hSMul n b))
  -/
  rw [add_comm, map_add_nsmul]
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_nat_add' [AddCommMonoidWithOne G] [AddMonoid H] [AddConstMapClass F G H 1 b]
    (f : F) (n : ℕ) (x : G) : f (↑n + x) = f x + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    b : H
    inst✝² : AddCommMonoidWithOne G
    inst✝¹ : AddMonoid H
    inst✝ : AddConstMapClass F G H 1 b
    f : F
    n : Nat
    x : G
    ⊢ Eq (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f x) (HSMul.hSMul n b))
  -/
  simpa using map_nsmul_add f n x
  /-
    🎉 no goals
  -/


theorem map_ofNat_add' [AddCommMonoidWithOne G] [AddMonoid H] [AddConstMapClass F G H 1 b]
    (f : F) (n : ℕ) [n.AtLeastTwo] (x : G) :
    f (OfNat.ofNat n + x) = f x + OfNat.ofNat n • b :=
  map_nat_add' f n x


theorem map_nat_add [AddCommMonoidWithOne G] [AddMonoidWithOne H] [AddConstMapClass F G H 1 1]
                                                         /-
                                                           F : Type u_1
                                                           G : Type u_2
                                                           H : Type u_3
                                                           inst✝³ : FunLike F G H
                                                           inst✝² : AddCommMonoidWithOne G
                                                           inst✝¹ : AddMonoidWithOne H
                                                           inst✝ : AddConstMapClass F G H 1 1
                                                           f : F
                                                           n : Nat
                                                           x : G
                                                           ⊢ Eq (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f x) ↑n)
                                                         -/
    (f : F) (n : ℕ) (x : G) : f (↑n + x) = f x + n := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem map_ofNat_add [AddCommMonoidWithOne G] [AddMonoidWithOne H] [AddConstMapClass F G H 1 1]
    (f : F) (n : ℕ) [n.AtLeastTwo] (x : G) :
    f (OfNat.ofNat n + x) = f x + OfNat.ofNat n :=
  map_nat_add f n x


@[scoped simp]
theorem map_sub_nsmul [AddGroup G] [AddGroup H] [AddConstMapClass F G H a b]
    (f : F) (x : G) (n : ℕ) : f (x - n • a) = f x - n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddGroup G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H a b
    f : F
    x : G
    n : Nat
    ⊢ Eq (f (HSub.hSub x (HSMul.hSMul n a))) (HSub.hSub (f x) (HSMul.hSMul n b))
  -/
  conv_rhs => rw [← sub_add_cancel x (n • a), map_add_nsmul, add_sub_cancel_right]
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_sub_const [AddGroup G] [AddGroup H] [AddConstMapClass F G H a b]
    (f : F) (x : G) : f (x - a) = f x - b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddGroup G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H a b
    f : F
    x : G
    ⊢ Eq (f (HSub.hSub x a)) (HSub.hSub (f x) b)
  -/
  simpa using map_sub_nsmul f x 1
  /-
    🎉 no goals
  -/


theorem map_sub_one [AddGroup G] [One G] [AddGroup H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) : f (x - 1) = f x - b :=
  map_sub_const f x


@[scoped simp]
theorem map_sub_nat' [AddGroupWithOne G] [AddGroup H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) (n : ℕ) : f (x - n) = f x - n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    b : H
    inst✝² : AddGroupWithOne G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H 1 b
    f : F
    x : G
    n : Nat
    ⊢ Eq (f (HSub.hSub x ↑n)) (HSub.hSub (f x) (HSMul.hSMul n b))
  -/
  simpa using map_sub_nsmul f x n
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_sub_ofNat' [AddGroupWithOne G] [AddGroup H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) (n : ℕ) [n.AtLeastTwo] :
    f (x - no_index (OfNat.ofNat n)) = f x - OfNat.ofNat n • b :=
  map_sub_nat' f x n


@[scoped simp]
theorem map_add_zsmul [AddGroup G] [AddGroup H] [AddConstMapClass F G H a b]
    (f : F) (x : G) : ∀ n : ℤ, f (x + n • a) = f x + n • b
                  /-
                    F : Type u_1
                    G : Type u_2
                    H : Type u_3
                    inst✝³ : FunLike F G H
                    a : G
                    b : H
                    inst✝² : AddGroup G
                    inst✝¹ : AddGroup H
                    inst✝ : AddConstMapClass F G H a b
                    f : F
                    x : G
                    n : Nat
                    ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul (↑n) a))) (HAdd.hAdd (f x) (HSMul.hSMul (↑n) …
                  -/
  | (n : ℕ) => by simp
                  /-
                    🎉 no goals
                  -/
                     /-
                       F : Type u_1
                       G : Type u_2
                       H : Type u_3
                       inst✝³ : FunLike F G H
                       a : G
                       b : H
                       inst✝² : AddGroup G
                       inst✝¹ : AddGroup H
                       inst✝ : AddConstMapClass F G H a b
                       f : F
                       x : G
                       n : Nat
                       ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul (Int.negSucc n) a))) (HAdd.hAdd (f x) (HSMul …
                     -/
  | .negSucc n => by simp [← sub_eq_add_neg]
                     /-
                       🎉 no goals
                     -/


@[scoped simp]
theorem map_zsmul_const [AddGroup G] [AddGroup H] [AddConstMapClass F G H a b]
    (f : F) (n : ℤ) : f (n • a) = f 0 + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddGroup G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H a b
    f : F
    n : Int
    ⊢ Eq (f (HSMul.hSMul n a)) (HAdd.hAdd (f 0) (HSMul.hSMul n b))
  -/
  simpa using map_add_zsmul f 0 n
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_add_int' [AddGroupWithOne G] [AddGroup H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) (n : ℤ) : f (x + n) = f x + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    b : H
    inst✝² : AddGroupWithOne G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H 1 b
    f : F
    x : G
    n : Int
    ⊢ Eq (f (HAdd.hAdd x ↑n)) (HAdd.hAdd (f x) (HSMul.hSMul n b))
  -/
  rw [← map_add_zsmul f x n, zsmul_one]
  /-
    🎉 no goals
  -/


theorem map_add_int [AddGroupWithOne G] [AddGroupWithOne H] [AddConstMapClass F G H 1 1]
                                                        /-
                                                          F : Type u_1
                                                          G : Type u_2
                                                          H : Type u_3
                                                          inst✝³ : FunLike F G H
                                                          inst✝² : AddGroupWithOne G
                                                          inst✝¹ : AddGroupWithOne H
                                                          inst✝ : AddConstMapClass F G H 1 1
                                                          f : F
                                                          x : G
                                                          n : Int
                                                          ⊢ Eq (f (HAdd.hAdd x ↑n)) (HAdd.hAdd (f x) ↑n)
                                                        -/
    (f : F) (x : G) (n : ℤ) : f (x + n) = f x + n := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


@[scoped simp]
theorem map_sub_zsmul [AddGroup G] [AddGroup H] [AddConstMapClass F G H a b]
    (f : F) (x : G) (n : ℤ) : f (x - n • a) = f x - n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddGroup G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H a b
    f : F
    x : G
    n : Int
    ⊢ Eq (f (HSub.hSub x (HSMul.hSMul n a))) (HSub.hSub (f x) (HSMul.hSMul n b))
  -/
  simpa [sub_eq_add_neg] using map_add_zsmul f x (-n)
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_sub_int' [AddGroupWithOne G] [AddGroup H] [AddConstMapClass F G H 1 b]
    (f : F) (x : G) (n : ℤ) : f (x - n) = f x - n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    b : H
    inst✝² : AddGroupWithOne G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H 1 b
    f : F
    x : G
    n : Int
    ⊢ Eq (f (HSub.hSub x ↑n)) (HSub.hSub (f x) (HSMul.hSMul n b))
  -/
  rw [← map_sub_zsmul, zsmul_one]
  /-
    🎉 no goals
  -/


theorem map_sub_int [AddGroupWithOne G] [AddGroupWithOne H] [AddConstMapClass F G H 1 1]
                                                        /-
                                                          F : Type u_1
                                                          G : Type u_2
                                                          H : Type u_3
                                                          inst✝³ : FunLike F G H
                                                          inst✝² : AddGroupWithOne G
                                                          inst✝¹ : AddGroupWithOne H
                                                          inst✝ : AddConstMapClass F G H 1 1
                                                          f : F
                                                          x : G
                                                          n : Int
                                                          ⊢ Eq (f (HSub.hSub x ↑n)) (HSub.hSub (f x) ↑n)
                                                        -/
    (f : F) (x : G) (n : ℤ) : f (x - n) = f x - n := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


@[scoped simp]
theorem map_zsmul_add [AddCommGroup G] [AddGroup H] [AddConstMapClass F G H a b]
    (f : F) (n : ℤ) (x : G) : f (n • a + x) = f x + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    a : G
    b : H
    inst✝² : AddCommGroup G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H a b
    f : F
    n : Int
    x : G
    ⊢ Eq (f (HAdd.hAdd (HSMul.hSMul n a) x)) (HAdd.hAdd (f x) (HSMul.hSMul n b))
  -/
  rw [add_comm, map_add_zsmul]
  /-
    🎉 no goals
  -/


@[scoped simp]
theorem map_int_add' [AddCommGroupWithOne G] [AddGroup H] [AddConstMapClass F G H 1 b]
    (f : F) (n : ℤ) (x : G) : f (↑n + x) = f x + n • b := by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝³ : FunLike F G H
    b : H
    inst✝² : AddCommGroupWithOne G
    inst✝¹ : AddGroup H
    inst✝ : AddConstMapClass F G H 1 b
    f : F
    n : Int
    x : G
    ⊢ Eq (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f x) (HSMul.hSMul n b))
  -/
  rw [← map_zsmul_add, zsmul_one]
  /-
    🎉 no goals
  -/


theorem map_int_add [AddCommGroupWithOne G] [AddGroupWithOne H] [AddConstMapClass F G H 1 1]
                                                         /-
                                                           F : Type u_1
                                                           G : Type u_2
                                                           H : Type u_3
                                                           inst✝³ : FunLike F G H
                                                           inst✝² : AddCommGroupWithOne G
                                                           inst✝¹ : AddGroupWithOne H
                                                           inst✝ : AddConstMapClass F G H 1 1
                                                           f : F
                                                           n : Int
                                                           x : G
                                                           ⊢ Eq (f (HAdd.hAdd (↑n) x)) (HAdd.hAdd (f x) ↑n)
                                                         -/
    (f : F) (n : ℤ) (x : G) : f (↑n + x) = f x + n := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem map_fract {R : Type*} [LinearOrderedRing R] [FloorRing R] [AddGroup H]
    [FunLike F R H] [AddConstMapClass F R H 1 b] (f : F) (x : R) :
    f (Int.fract x) = f x - ⌊x⌋ • b :=
  map_sub_int' ..


/-- Auxiliary lemmas for the "monotonicity on a fundamental interval implies monotonicity" lemmas.
We formulate it for any relation so that the proof works both for `Monotone` and `StrictMono`. -/
protected theorem rel_map_of_Icc [LinearOrderedAddCommGroup G] [Archimedean G] [AddGroup H]
    [AddConstMapClass F G H a b] {f : F} {R : H → H → Prop} [IsTrans H R]
    [hR : CovariantClass H H (fun x y ↦ y + x) R] (ha : 0 < a) {l : G}
    (hf : ∀ x ∈ Icc l (l + a), ∀ y ∈ Icc l (l + a), x < y → R (f x) (f y)) :
    ((· < ·) ⇒ R) f f := fun x y hxy ↦ by
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝⁵ : FunLike F G H
    a : G
    b : H
    inst✝⁴ : LinearOrderedAddCommGroup G
    inst✝³ : Archimedean G
    inst✝² : AddGroup H
    inst✝¹ : AddConstMapClass F G H a b
    f : F
    R : H → H → Prop
    inst✝ : IsTrans H R
    hR : CovariantClass H H (fun x y => HAdd.hAdd y x) R
    ha : LT.lt 0 a
    l : G
    hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
    x y : G
    hxy : (fun x1 x2 => LT.lt x1 x2) x y
    ⊢ R (f x) (f y)
  -/
  replace hR := hR.elim
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝⁵ : FunLike F G H
    a : G
    b : H
    inst✝⁴ : LinearOrderedAddCommGroup G
    inst✝³ : Archimedean G
    inst✝² : AddGroup H
    inst✝¹ : AddConstMapClass F G H a b
    f : F
    R : H → H → Prop
    inst✝ : IsTrans H R
    ha : LT.lt 0 a
    l : G
    hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
    x y : G
    hxy : (fun x1 x2 => LT.lt x1 x2) x y
    hR : Covariant H H (fun x y => HAdd.hAdd y x) R
    ⊢ R (f x) (f y)
  -/
  have ha' : 0 ≤ a := ha.le
  -- Shift both points by `m • a` so that `l ≤ x < l + a`
  /-
    F : Type u_1
    G : Type u_2
    H : Type u_3
    inst✝⁵ : FunLike F G H
    a : G
    b : H
    inst✝⁴ : LinearOrderedAddCommGroup G
    inst✝³ : Archimedean G
    inst✝² : AddGroup H
    inst✝¹ : AddConstMapClass F G H a b
    f : F
    R : H → H → Prop
    inst✝ : IsTrans H R
    ha : LT.lt 0 a
    l : G
    hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
    x y : G
    hxy : (fun x1 x2 => LT.lt x1 x2) x y
    hR : Covariant H H (fun x y => HAdd.hAdd y x) R
    ha' : LE.le 0 a
    ⊢ R (f x) (f y)
  -/
  wlog hx : x ∈ Ico l (l + a) generalizing x y
    /-
      case inr
      F : Type u_1
      G : Type u_2
      H : Type u_3
      inst✝⁵ : FunLike F G H
      a : G
      b : H
      inst✝⁴ : LinearOrderedAddCommGroup G
      inst✝³ : Archimedean G
      inst✝² : AddGroup H
      inst✝¹ : AddConstMapClass F G H a b
      f : F
      R : H → H → Prop
      inst✝ : IsTrans H R
      ha : LT.lt 0 a
      l : G
      hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
      x y : G
      hxy : (fun x1 x2 => LT.lt x1 x2) x y
      hR : Covariant H H (fun x y => HAdd.hAdd y x) R
      ha' : LE.le 0 a
      this : ∀ (x y : G), LT.lt x y → Membership.mem (Set.Ico l (HAdd.hAdd l a)) x → …
      hx : Not (Membership.mem (Set.Ico l (HAdd.hAdd l a)) x)
      ⊢ R (f x) (f y)
    -/
  · rcases existsUnique_sub_zsmul_mem_Ico ha x l with ⟨m, hm, -⟩
    /-
      case inr.intro.intro
      F : Type u_1
      G : Type u_2
      H : Type u_3
      inst✝⁵ : FunLike F G H
      a : G
      b : H
      inst✝⁴ : LinearOrderedAddCommGroup G
      inst✝³ : Archimedean G
      inst✝² : AddGroup H
      inst✝¹ : AddConstMapClass F G H a b
      f : F
      R : H → H → Prop
      inst✝ : IsTrans H R
      ha : LT.lt 0 a
      l : G
      hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
      x y : G
      hxy : (fun x1 x2 => LT.lt x1 x2) x y
      hR : Covariant H H (fun x y => HAdd.hAdd y x) R
      ha' : LE.le 0 a
      this : ∀ (x y : G), LT.lt x y → Membership.mem (Set.Ico l (HAdd.hAdd l a)) x → …
      hx : Not (Membership.mem (Set.Ico l (HAdd.hAdd l a)) x)
      m : Int
      hm : Membership.mem (Set.Ico l (HAdd.hAdd l a)) (HSub.hSub x (HSMul.hSMul m a))
      ⊢ R (f x) (f y)
    -/
    suffices R (f (x - m • a)) (f (y - m • a)) by simpa using hR (m • b) this
    /-
      case inr.intro.intro
      F : Type u_1
      G : Type u_2
      H : Type u_3
      inst✝⁵ : FunLike F G H
      a : G
      b : H
      inst✝⁴ : LinearOrderedAddCommGroup G
      inst✝³ : Archimedean G
      inst✝² : AddGroup H
      inst✝¹ : AddConstMapClass F G H a b
      f : F
      R : H → H → Prop
      inst✝ : IsTrans H R
      ha : LT.lt 0 a
      l : G
      hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
      x y : G
      hxy : (fun x1 x2 => LT.lt x1 x2) x y
      hR : Covariant H H (fun x y => HAdd.hAdd y x) R
      ha' : LE.le 0 a
      this : ∀ (x y : G), LT.lt x y → Membership.mem (Set.Ico l (HAdd.hAdd l a)) x → …
      hx : Not (Membership.mem (Set.Ico l (HAdd.hAdd l a)) x)
      m : Int
      hm : Membership.mem (Set.Ico l (HAdd.hAdd l a)) (HSub.hSub x (HSMul.hSMul m a))
      ⊢ R (f (HSub.hSub x (HSMul.hSMul m a))) (f (HSub.hSub y (HSMul.hSMul m a)))
    -/
    exact this _ _ (by simpa) hm
    /-
      🎉 no goals
    -/
  · -- Now find `n` such that `l + n • a < y ≤ l + (n + 1) • a`
    /-
      F : Type u_1
      G : Type u_2
      H : Type u_3
      inst✝⁵ : FunLike F G H
      a : G
      b : H
      inst✝⁴ : LinearOrderedAddCommGroup G
      inst✝³ : Archimedean G
      inst✝² : AddGroup H
      inst✝¹ : AddConstMapClass F G H a b
      f : F
      R : H → H → Prop
      inst✝ : IsTrans H R
      ha : LT.lt 0 a
      l : G
      hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
      hR : Covariant H H (fun x y => HAdd.hAdd y x) R
      ha' : LE.le 0 a
      x y : G
      hxy : LT.lt x y
      hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
      ⊢ R (f x) (f y)
    -/
    rcases existsUnique_sub_zsmul_mem_Ioc ha y l with ⟨n, hny, -⟩
    /-
      case intro.intro
      F : Type u_1
      G : Type u_2
      H : Type u_3
      inst✝⁵ : FunLike F G H
      a : G
      b : H
      inst✝⁴ : LinearOrderedAddCommGroup G
      inst✝³ : Archimedean G
      inst✝² : AddGroup H
      inst✝¹ : AddConstMapClass F G H a b
      f : F
      R : H → H → Prop
      inst✝ : IsTrans H R
      ha : LT.lt 0 a
      l : G
      hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
      hR : Covariant H H (fun x y => HAdd.hAdd y x) R
      ha' : LE.le 0 a
      x y : G
      hxy : LT.lt x y
      hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
      n : Int
      hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
      ⊢ R (f x) (f y)
    -/
    rcases lt_trichotomy n 0 with hn | rfl | hn
    · -- Since `l ≤ x ≤ y`, the case `n < 0` is impossible
      /-
        case intro.intro.inl
        F : Type u_1
        G : Type u_2
        H : Type u_3
        inst✝⁵ : FunLike F G H
        a : G
        b : H
        inst✝⁴ : LinearOrderedAddCommGroup G
        inst✝³ : Archimedean G
        inst✝² : AddGroup H
        inst✝¹ : AddConstMapClass F G H a b
        f : F
        R : H → H → Prop
        inst✝ : IsTrans H R
        ha : LT.lt 0 a
        l : G
        hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
        hR : Covariant H H (fun x y => HAdd.hAdd y x) R
        ha' : LE.le 0 a
        x y : G
        hxy : LT.lt x y
        hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
        n : Int
        hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
        hn : LT.lt n 0
        ⊢ R (f x) (f y)
      -/
      refine absurd ?_ hxy.not_le
      calc
        y ≤ l + a + n • a := sub_le_iff_le_add.1 hny.2
        _ = l + (n + 1) • a := by rw [add_comm n, add_smul, one_smul, add_assoc]
        _ ≤ l + 0 • a := add_le_add_left (zsmul_le_zsmul_left ha.le (by omega)) _
        _ ≤ x := by simpa using hx.1
    · -- If `n = 0`, then `l < y ≤ l + a`, hence we can apply the assumption
      /-
        case intro.intro.inr.inl
        F : Type u_1
        G : Type u_2
        H : Type u_3
        inst✝⁵ : FunLike F G H
        a : G
        b : H
        inst✝⁴ : LinearOrderedAddCommGroup G
        inst✝³ : Archimedean G
        inst✝² : AddGroup H
        inst✝¹ : AddConstMapClass F G H a b
        f : F
        R : H → H → Prop
        inst✝ : IsTrans H R
        ha : LT.lt 0 a
        l : G
        hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
        hR : Covariant H H (fun x y => HAdd.hAdd y x) R
        ha' : LE.le 0 a
        x y : G
        hxy : LT.lt x y
        hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
        hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul 0 a))
        ⊢ R (f x) (f y)
      -/
      exact hf x (Ico_subset_Icc_self hx) y (by simpa using Ioc_subset_Icc_self hny) hxy
      /-
        🎉 no goals
      -/
    · -- In the remaining case `0 < n` we use transitivity.
      -- If `R = (· < ·)`, then the proof looks like
      -- `f x < f (l + a) ≤ f (l + n • a) < f y`
      /-
        case intro.intro.inr.inr
        F : Type u_1
        G : Type u_2
        H : Type u_3
        inst✝⁵ : FunLike F G H
        a : G
        b : H
        inst✝⁴ : LinearOrderedAddCommGroup G
        inst✝³ : Archimedean G
        inst✝² : AddGroup H
        inst✝¹ : AddConstMapClass F G H a b
        f : F
        R : H → H → Prop
        inst✝ : IsTrans H R
        ha : LT.lt 0 a
        l : G
        hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
        hR : Covariant H H (fun x y => HAdd.hAdd y x) R
        ha' : LE.le 0 a
        x y : G
        hxy : LT.lt x y
        hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
        n : Int
        hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
        hn : LT.lt 0 n
        ⊢ R (f x) (f y)
      -/
      trans f (l + (1 : ℤ) • a)
        /-
          F : Type u_1
          G : Type u_2
          H : Type u_3
          inst✝⁵ : FunLike F G H
          a : G
          b : H
          inst✝⁴ : LinearOrderedAddCommGroup G
          inst✝³ : Archimedean G
          inst✝² : AddGroup H
          inst✝¹ : AddConstMapClass F G H a b
          f : F
          R : H → H → Prop
          inst✝ : IsTrans H R
          ha : LT.lt 0 a
          l : G
          hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
          hR : Covariant H H (fun x y => HAdd.hAdd y x) R
          ha' : LE.le 0 a
          x y : G
          hxy : LT.lt x y
          hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
          n : Int
          hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
          hn : LT.lt 0 n
          ⊢ R (f x) (f (HAdd.hAdd l (HSMul.hSMul 1 a)))
        -/
      · rw [one_zsmul]
        /-
          F : Type u_1
          G : Type u_2
          H : Type u_3
          inst✝⁵ : FunLike F G H
          a : G
          b : H
          inst✝⁴ : LinearOrderedAddCommGroup G
          inst✝³ : Archimedean G
          inst✝² : AddGroup H
          inst✝¹ : AddConstMapClass F G H a b
          f : F
          R : H → H → Prop
          inst✝ : IsTrans H R
          ha : LT.lt 0 a
          l : G
          hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
          hR : Covariant H H (fun x y => HAdd.hAdd y x) R
          ha' : LE.le 0 a
          x y : G
          hxy : LT.lt x y
          hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
          n : Int
          hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
          hn : LT.lt 0 n
          ⊢ R (f x) (f (HAdd.hAdd l a))
        -/
        exact hf x (Ico_subset_Icc_self hx) (l + a) (by simpa) hx.2
        /-
          🎉 no goals
        -/
      have hy : R (f (l + n • a)) (f y) := by
        rw [← sub_add_cancel y (n • a), map_add_zsmul, map_add_zsmul]
        refine hR _ <| hf _ ?_ _ (Ioc_subset_Icc_self hny) hny.1; simpa
      /-
        F : Type u_1
        G : Type u_2
        H : Type u_3
        inst✝⁵ : FunLike F G H
        a : G
        b : H
        inst✝⁴ : LinearOrderedAddCommGroup G
        inst✝³ : Archimedean G
        inst✝² : AddGroup H
        inst✝¹ : AddConstMapClass F G H a b
        f : F
        R : H → H → Prop
        inst✝ : IsTrans H R
        ha : LT.lt 0 a
        l : G
        hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
        hR : Covariant H H (fun x y => HAdd.hAdd y x) R
        ha' : LE.le 0 a
        x y : G
        hxy : LT.lt x y
        hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
        n : Int
        hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
        hn : LT.lt 0 n
        hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
        ⊢ R (f (HAdd.hAdd l (HSMul.hSMul 1 a))) (f y)
      -/
      rw [← Int.add_one_le_iff, zero_add] at hn
      /-
        F : Type u_1
        G : Type u_2
        H : Type u_3
        inst✝⁵ : FunLike F G H
        a : G
        b : H
        inst✝⁴ : LinearOrderedAddCommGroup G
        inst✝³ : Archimedean G
        inst✝² : AddGroup H
        inst✝¹ : AddConstMapClass F G H a b
        f : F
        R : H → H → Prop
        inst✝ : IsTrans H R
        ha : LT.lt 0 a
        l : G
        hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
        hR : Covariant H H (fun x y => HAdd.hAdd y x) R
        ha' : LE.le 0 a
        x y : G
        hxy : LT.lt x y
        hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
        n : Int
        hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
        hn : LE.le 1 n
        hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
        ⊢ R (f (HAdd.hAdd l (HSMul.hSMul 1 a))) (f y)
      -/
      rcases hn.eq_or_lt with rfl | hn; · assumption
                                          /-
                                            🎉 no goals
                                          -/
      /-
        case inr
        F : Type u_1
        G : Type u_2
        H : Type u_3
        inst✝⁵ : FunLike F G H
        a : G
        b : H
        inst✝⁴ : LinearOrderedAddCommGroup G
        inst✝³ : Archimedean G
        inst✝² : AddGroup H
        inst✝¹ : AddConstMapClass F G H a b
        f : F
        R : H → H → Prop
        inst✝ : IsTrans H R
        ha : LT.lt 0 a
        l : G
        hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
        hR : Covariant H H (fun x y => HAdd.hAdd y x) R
        ha' : LE.le 0 a
        x y : G
        hxy : LT.lt x y
        hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
        n : Int
        hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
        hn✝ : LE.le 1 n
        hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
        hn : LT.lt 1 n
        ⊢ R (f (HAdd.hAdd l (HSMul.hSMul 1 a))) (f y)
      -/
      trans f (l + n • a)
        /-
          F : Type u_1
          G : Type u_2
          H : Type u_3
          inst✝⁵ : FunLike F G H
          a : G
          b : H
          inst✝⁴ : LinearOrderedAddCommGroup G
          inst✝³ : Archimedean G
          inst✝² : AddGroup H
          inst✝¹ : AddConstMapClass F G H a b
          f : F
          R : H → H → Prop
          inst✝ : IsTrans H R
          ha : LT.lt 0 a
          l : G
          hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
          hR : Covariant H H (fun x y => HAdd.hAdd y x) R
          ha' : LE.le 0 a
          x y : G
          hxy : LT.lt x y
          hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
          n : Int
          hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
          hn✝ : LE.le 1 n
          hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
          hn : LT.lt 1 n
          ⊢ R (f (HAdd.hAdd l (HSMul.hSMul 1 a))) (f (HAdd.hAdd l (HSMul.hSMul n a)))
        -/
      · refine Int.rel_of_forall_rel_succ_of_lt R (f := (f <| l + · • a)) (fun k ↦ ?_) hn
        /-
          F : Type u_1
          G : Type u_2
          H : Type u_3
          inst✝⁵ : FunLike F G H
          a : G
          b : H
          inst✝⁴ : LinearOrderedAddCommGroup G
          inst✝³ : Archimedean G
          inst✝² : AddGroup H
          inst✝¹ : AddConstMapClass F G H a b
          f : F
          R : H → H → Prop
          inst✝ : IsTrans H R
          ha : LT.lt 0 a
          l : G
          hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
          hR : Covariant H H (fun x y => HAdd.hAdd y x) R
          ha' : LE.le 0 a
          x y : G
          hxy : LT.lt x y
          hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
          n : Int
          hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
          hn✝ : LE.le 1 n
          hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
          hn : LT.lt 1 n
          k : Int
          ⊢ R ((fun x => f (HAdd.hAdd l (HSMul.hSMul x a))) k) ((fun x => f (HAdd.hAdd l …
        -/
        simp_rw [add_comm k 1, add_zsmul, ← add_assoc, one_zsmul, map_add_zsmul]
        /-
          F : Type u_1
          G : Type u_2
          H : Type u_3
          inst✝⁵ : FunLike F G H
          a : G
          b : H
          inst✝⁴ : LinearOrderedAddCommGroup G
          inst✝³ : Archimedean G
          inst✝² : AddGroup H
          inst✝¹ : AddConstMapClass F G H a b
          f : F
          R : H → H → Prop
          inst✝ : IsTrans H R
          ha : LT.lt 0 a
          l : G
          hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
          hR : Covariant H H (fun x y => HAdd.hAdd y x) R
          ha' : LE.le 0 a
          x y : G
          hxy : LT.lt x y
          hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
          n : Int
          hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
          hn✝ : LE.le 1 n
          hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
          hn : LT.lt 1 n
          k : Int
          ⊢ R (HAdd.hAdd (f l) (HSMul.hSMul k b)) (HAdd.hAdd (f (HAdd.hAdd l a)) (HSMul. …
        -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
        refine hR (k • b) (hf _ ?_ _ ?_ ?_) <;> simpa
                                                /-
                                                  🎉 no goals
                                                -/
        /-
          F : Type u_1
          G : Type u_2
          H : Type u_3
          inst✝⁵ : FunLike F G H
          a : G
          b : H
          inst✝⁴ : LinearOrderedAddCommGroup G
          inst✝³ : Archimedean G
          inst✝² : AddGroup H
          inst✝¹ : AddConstMapClass F G H a b
          f : F
          R : H → H → Prop
          inst✝ : IsTrans H R
          ha : LT.lt 0 a
          l : G
          hf : ∀ (x : G), Membership.mem (Set.Icc l (HAdd.hAdd l a)) x → ∀ (y : G), Memb …
          hR : Covariant H H (fun x y => HAdd.hAdd y x) R
          ha' : LE.le 0 a
          x y : G
          hxy : LT.lt x y
          hx : Membership.mem (Set.Ico l (HAdd.hAdd l a)) x
          n : Int
          hny : Membership.mem (Set.Ioc l (HAdd.hAdd l a)) (HSub.hSub y (HSMul.hSMul n a))
          hn✝ : LE.le 1 n
          hy : R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
          hn : LT.lt 1 n
          ⊢ R (f (HAdd.hAdd l (HSMul.hSMul n a))) (f y)
        -/
      · assumption
        /-
          🎉 no goals
        -/


theorem monotone_iff_Icc [LinearOrderedAddCommGroup G] [Archimedean G] [OrderedAddCommGroup H]
    [AddConstMapClass F G H a b] {f : F} (ha : 0 < a) (l : G) :
    Monotone f ↔ MonotoneOn f (Icc l (l + a)) :=
  ⟨(Monotone.monotoneOn · _), fun hf ↦ monotone_iff_forall_lt.2 <|
    AddConstMapClass.rel_map_of_Icc ha fun _x hx _y hy hxy ↦ hf hx hy hxy.le⟩


theorem antitone_iff_Icc [LinearOrderedAddCommGroup G] [Archimedean G] [OrderedAddCommGroup H]
    [AddConstMapClass F G H a b] {f : F} (ha : 0 < a) (l : G) :
    Antitone f ↔ AntitoneOn f (Icc l (l + a)) :=
  monotone_iff_Icc (H := Hᵒᵈ) ha l


theorem strictMono_iff_Icc [LinearOrderedAddCommGroup G] [Archimedean G] [OrderedAddCommGroup H]
    [AddConstMapClass F G H a b] {f : F} (ha : 0 < a) (l : G) :
    StrictMono f ↔ StrictMonoOn f (Icc l (l + a)) :=
  ⟨(StrictMono.strictMonoOn · _), AddConstMapClass.rel_map_of_Icc ha⟩


theorem strictAnti_iff_Icc [LinearOrderedAddCommGroup G] [Archimedean G] [OrderedAddCommGroup H]
    [AddConstMapClass F G H a b] {f : F} (ha : 0 < a) (l : G) :
    StrictAnti f ↔ StrictAntiOn f (Icc l (l + a)) :=
  strictMono_iff_Icc (H := Hᵒᵈ) ha l


instance : FunLike (G →+c[a, b] H) G H where
  coe := AddConstMap.toFun
  coe_injective' | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


@[simp, push_cast] theorem coe_mk (f : G → H) (hf) : ⇑(mk f hf : G →+c[a, b] H) = f := rfl

@[simp] theorem mk_coe (f : G →+c[a, b] H) : mk f f.2 = f := rfl

@[simp] theorem toFun_eq_coe (f : G →+c[a, b] H) : f.toFun = f := rfl


instance : AddConstMapClass (G →+c[a, b] H) G H a b where
  map_add_const f := f.map_add_const'


@[ext] protected theorem ext {f g : G →+c[a, b] H} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


/-- The identity map as `G →+c[a, a] G`. -/
@[simps (config := .asFn)]
protected def id : G →+c[a, a] G := ⟨id, fun _ ↦ rfl⟩


instance : Inhabited (G →+c[a, a] G) := ⟨.id⟩


/-- Composition of two `AddConstMap`s. -/
@[simps (config := .asFn)]
def comp {K : Type*} [Add K] {c : K} (g : H →+c[b, c] K) (f : G →+c[a, b] H) :
    G →+c[a, c] K :=
             /-
               G : Type u_1
               H : Type u_2
               inst✝² : Add G
               inst✝¹ : Add H
               a : G
               b : H
               K : Type u_3
               inst✝ : Add K
               c : K
               g : AddConstMap H K b c
               f : AddConstMap G H a b
               ⊢ ∀ (x : G), Eq (Function.comp (⇑g) (⇑f) (HAdd.hAdd x a)) (HAdd.hAdd (Function …
             -/
  ⟨g ∘ f, by simp⟩
             /-
               🎉 no goals
             -/


@[simp] theorem comp_id (f : G →+c[a, b] H) : f.comp .id = f := rfl

@[simp] theorem id_comp (f : G →+c[a, b] H) : .comp .id f = f := rfl


/-- Change constants `a` and `b` in `(f : G →+c[a, b] H)` to improve definitional equalities. -/
@[simps (config := .asFn)]
def replaceConsts (f : G →+c[a, b] H) (a' b') (ha : a = a') (hb : b = b') :
    G →+c[a', b'] H where
  toFun := f
  map_add_const' := ha ▸ hb ▸ f.map_add_const'


/-- If `f` is an `AddConstMap`, then so is `(c +ᵥ f ·)`. -/
instance {K : Type*} [VAdd K H] [VAddAssocClass K H H] : VAdd K (G →+c[a, b] H) :=
                                  /-
                                    G : Type u_1
                                    H : Type u_2
                                    inst✝³ : Add G
                                    inst✝² : Add H
                                    a : G
                                    b : H
                                    K : Type u_3
                                    inst✝¹ : VAdd K H
                                    inst✝ : VAddAssocClass K H H
                                    c : K
                                    f : AddConstMap G H a b
                                    x : G
                                    ⊢ Eq (HVAdd.hVAdd c (⇑f) (HAdd.hAdd x a)) (HAdd.hAdd (HVAdd.hVAdd c (⇑f) x) b)
                                  -/
  ⟨fun c f ↦ ⟨c +ᵥ ⇑f, fun x ↦ by simp [vadd_add_assoc]⟩⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp, norm_cast]
theorem coe_vadd {K : Type*} [VAdd K H] [VAddAssocClass K H H] (c : K) (f : G →+c[a, b] H) :
    ⇑(c +ᵥ f) = c +ᵥ ⇑f :=
  rfl


instance {K : Type*} [AddMonoid K] [AddAction K H] [VAddAssocClass K H H] :
    AddAction K (G →+c[a, b] H) :=
  DFunLike.coe_injective.addAction _ coe_vadd


instance : Mul (G →+c[a, a] G) := ⟨comp⟩

instance : One (G →+c[a, a] G) := ⟨.id⟩


instance : Pow (G →+c[a, a] G) ℕ where
  pow f n := ⟨f^[n], Commute.iterate_left (AddConstMapClass.semiconj f) _⟩


instance : Monoid (G →+c[a, a] G) :=
  DFunLike.coe_injective.monoid (M₂ := Function.End G) _ rfl (fun _ _ ↦ rfl) fun _ _ ↦ rfl


theorem mul_def (f g : G →+c[a, a] G) : f * g = f.comp g := rfl

@[simp, push_cast] theorem coe_mul (f g : G →+c[a, a] G) : ⇑(f * g) = f ∘ g := rfl


theorem one_def : (1 : G →+c[a, a] G) = .id := rfl

@[simp, push_cast] theorem coe_one : ⇑(1 : G →+c[a, a] G) = id := rfl


@[simp, push_cast] theorem coe_pow (f : G →+c[a, a] G) (n : ℕ) : ⇑(f ^ n) = f^[n] := rfl


theorem pow_apply (f : G →+c[a, a] G) (n : ℕ) (x : G) : (f ^ n) x = f^[n] x := rfl


/-- Coercion to functions as a monoid homomorphism to `Function.End G`. -/
@[simps (config := .asFn)]
def toEnd : (G →+c[a, a] G) →* Function.End G where
  toFun := DFunLike.coe
  map_mul' _ _ := rfl
  map_one' := rfl


/-- Pointwise scalar multiplication of `f : G →+c[a, b] H` as a map `G →+c[a, c • b] H`. -/
@[simps (config := .asFn)]
def smul [DistribSMul K H] (c : K) (f : G →+c[a, b] H) : G →+c[a, c • b] H where
  toFun := c • ⇑f
                         /-
                           G : Type u_1
                           H : Type u_2
                           K : Type u_3
                           inst✝² : Add G
                           inst✝¹ : AddZeroClass H
                           a : G
                           b : H
                           inst✝ : DistribSMul K H
                           c : K
                           f : AddConstMap G H a b
                           x : G
                           ⊢ Eq (HSMul.hSMul c (⇑f) (HAdd.hAdd x a)) (HAdd.hAdd (HSMul.hSMul c (⇑f) x) (H …
                         -/
  map_add_const' x := by simp [smul_add]
                         /-
                           🎉 no goals
                         -/


/-- The map that sends `c` to a translation by `c`
as a monoid homomorphism from `Multiplicative G` to `G →+c[a, a] G`. -/
@[simps! (config := .asFn)]
def addLeftHom : Multiplicative G →* (G →+c[a, a] G) where
  toFun c := c.toAdd +ᵥ .id
                 /-
                   G : Type u_1
                   inst✝ : AddMonoid G
                   a : G
                   ⊢ Eq ((fun c => HVAdd.hVAdd (Multiplicative.toAdd c) AddConstMap.id) 1) 1
                 -/
  map_one' := by ext; apply zero_add
                      /-
                        🎉 no goals
                      -/
                     /-
                       G : Type u_1
                       inst✝ : AddMonoid G
                       a : G
                       x✝¹ x✝ : Multiplicative G
                       ⊢ Eq ({ toFun := fun c => HVAdd.hVAdd (Multiplicative.toAdd c) AddConstMap.id, …
                     -/
  map_mul' _ _ := by ext; apply add_assoc
                          /-
                            🎉 no goals
                          -/


/-- If `f : G → H` is an `AddConstMap`, then so is `fun x ↦ -f (-x)`. -/
@[simps! apply_coe]
def conjNeg : (G →+c[a, b] H) ≃ (G →+c[a, b] H) :=
                                                           /-
                                                             G : Type u_1
                                                             H : Type u_2
                                                             inst✝¹ : AddCommGroup G
                                                             inst✝ : AddCommGroup H
                                                             a : G
                                                             b : H
                                                             f : AddConstMap G H a b
                                                             x✝ : G
                                                             ⊢ Eq ((fun x => Neg.neg (f (Neg.neg x))) (HAdd.hAdd x✝ a)) (HAdd.hAdd ((fun x  …
                                                           -/
  Involutive.toPerm (fun f ↦ ⟨fun x ↦ - f (-x), fun _ ↦ by simp [neg_add_eq_sub]⟩) fun _ ↦
                                                           /-
                                                             🎉 no goals
                                                           -/
                               /-
                                 G : Type u_1
                                 H : Type u_2
                                 inst✝¹ : AddCommGroup G
                                 inst✝ : AddCommGroup H
                                 a : G
                                 b : H
                                 x✝¹ : AddConstMap G H a b
                                 x✝ : G
                                 ⊢ Eq (((fun f => { toFun := fun x => Neg.neg (f (Neg.neg x)), map_add_const' : …
                               -/
    AddConstMap.ext fun _ ↦ by simp
                               /-
                                 🎉 no goals
                               -/


@[simp] theorem conjNeg_symm : (conjNeg (a := a) (b := b)).symm = conjNeg := rfl


/-- A map `f : R →+c[1, a] G` is defined by its values on `Set.Ico 0 1`. -/
def mkFract : (Ico (0 : R) 1 → G) ≃ (R →+c[1, a] G) where
  toFun f := ⟨fun x ↦ f ⟨Int.fract x, Int.fract_nonneg _, Int.fract_lt_one _⟩ + ⌊x⌋ • a, fun x ↦ by
    /-
      R : Type u_1
      G : Type u_2
      inst✝² : LinearOrderedRing R
      inst✝¹ : FloorRing R
      inst✝ : AddGroup G
      a : G
      f : ↑(Set.Ico 0 1) → G
      x : R
      ⊢ Eq ((fun x => HAdd.hAdd (f ⟨Int.fract x, ⋯⟩) (HSMul.hSMul (Int.floor x) a))  …
    -/
    simp [add_one_zsmul, add_assoc]⟩
    /-
      🎉 no goals
    -/
  invFun f x := f x
                   /-
                     R : Type u_1
                     G : Type u_2
                     inst✝² : LinearOrderedRing R
                     inst✝¹ : FloorRing R
                     inst✝ : AddGroup G
                     a : G
                     x✝ : ↑(Set.Ico 0 1) → G
                     ⊢ Eq ((fun f x => f ↑x) ((fun f => { toFun := fun x => HAdd.hAdd (f ⟨Int.fract …
                   -/
  left_inv _ := by ext x; simp [Int.fract_eq_self.2 x.2, Int.floor_eq_zero_iff.2 x.2]
                          /-
                            🎉 no goals
                          -/
                    /-
                      R : Type u_1
                      G : Type u_2
                      inst✝² : LinearOrderedRing R
                      inst✝¹ : FloorRing R
                      inst✝ : AddGroup G
                      a : G
                      f : AddConstMap R G 1 a
                      ⊢ Eq ((fun f => { toFun := fun x => HAdd.hAdd (f ⟨Int.fract x, ⋯⟩) (HSMul.hSMu …
                    -/
  right_inv f := by ext x; simp [map_fract]
                           /-
                             🎉 no goals
                           -/


