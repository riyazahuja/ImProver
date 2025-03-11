/-- A (non-commutative) Jordan multiplication. -/
class IsJordan [Mul A] : Prop where
  lmul_comm_rmul : ∀ a b : A, a * b * a = a * (b * a)
  lmul_lmul_comm_lmul : ∀ a b : A, a * a * (a * b) = a * (a * a * b)
  lmul_lmul_comm_rmul : ∀ a b : A, a * a * (b * a) = a * a * b * a
  lmul_comm_rmul_rmul : ∀ a b : A, a * b * (a * a) = a * (b * (a * a))
  rmul_comm_rmul_rmul : ∀ a b : A, b * a * (a * a) = b * (a * a) * a


/-- A commutative Jordan multiplication -/
class IsCommJordan [CommMagma A] : Prop where
  lmul_comm_rmul_rmul : ∀ a b : A, a * b * (a * a) = a * (b * (a * a))

-- see Note [lower instance priority]

/-- A (commutative) Jordan multiplication is also a Jordan multiplication -/
instance (priority := 100) IsCommJordan.toIsJordan [CommMagma A] [IsCommJordan A] : IsJordan A where
                           /-
                             A : Type u_1
                             inst✝¹ : CommMagma A
                             inst✝ : IsCommJordan A
                             a b : A
                             ⊢ Eq (HMul.hMul (HMul.hMul a b) a) (HMul.hMul a (HMul.hMul b a))
                           -/
  lmul_comm_rmul a b := by rw [mul_comm, mul_comm a b]
                           /-
                             🎉 no goals
                           -/
  lmul_lmul_comm_lmul a b := by
    rw [mul_comm (a * a) (a * b), IsCommJordan.lmul_comm_rmul_rmul,
      mul_comm b (a * a)]
  lmul_comm_rmul_rmul := IsCommJordan.lmul_comm_rmul_rmul
  lmul_lmul_comm_rmul a b := by
    rw [mul_comm (a * a) (b * a), mul_comm b a,
      IsCommJordan.lmul_comm_rmul_rmul, mul_comm, mul_comm b (a * a)]
  rmul_comm_rmul_rmul a b := by
    /-
      A : Type u_1
      inst✝¹ : CommMagma A
      inst✝ : IsCommJordan A
      a b : A
      ⊢ Eq (HMul.hMul (HMul.hMul b a) (HMul.hMul a a)) (HMul.hMul (HMul.hMul b (HMul …
    -/
    rw [mul_comm b a, IsCommJordan.lmul_comm_rmul_rmul, mul_comm]
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

/-- Semigroup multiplication satisfies the (non-commutative) Jordan axioms -/
instance (priority := 100) Semigroup.isJordan [Semigroup A] : IsJordan A where
                           /-
                             A : Type u_1
                             inst✝ : Semigroup A
                             a b : A
                             ⊢ Eq (HMul.hMul (HMul.hMul a b) a) (HMul.hMul a (HMul.hMul b a))
                           -/
  lmul_comm_rmul a b := by rw [mul_assoc]
                           /-
                             🎉 no goals
                           -/
                                /-
                                  A : Type u_1
                                  inst✝ : Semigroup A
                                  a b : A
                                  ⊢ Eq (HMul.hMul (HMul.hMul a a) (HMul.hMul a b)) (HMul.hMul a (HMul.hMul (HMul …
                                -/
  lmul_lmul_comm_lmul a b := by rw [mul_assoc, mul_assoc]
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  A : Type u_1
                                  inst✝ : Semigroup A
                                  a b : A
                                  ⊢ Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul a a)) (HMul.hMul a (HMul.hMul b (HM …
                                -/
                                /-
                                  A : Type u_1
                                  inst✝ : Semigroup A
                                  a b : A
                                  ⊢ Eq (HMul.hMul (HMul.hMul a a) (HMul.hMul b a)) (HMul.hMul (HMul.hMul (HMul.h …
                                -/
  lmul_comm_rmul_rmul a b := by rw [mul_assoc]
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  lmul_lmul_comm_rmul a b := by rw [← mul_assoc]
                                /-
                                  A : Type u_1
                                  inst✝ : Semigroup A
                                  a b : A
                                  ⊢ Eq (HMul.hMul (HMul.hMul b a) (HMul.hMul a a)) (HMul.hMul (HMul.hMul b (HMul …
                                -/
  rmul_comm_rmul_rmul a b := by rw [← mul_assoc, ← mul_assoc]
                                /-
                                  🎉 no goals
                                -/

-- see Note [lower instance priority]

instance (priority := 100) CommSemigroup.isCommJordan [CommSemigroup A] : IsCommJordan A where
  lmul_comm_rmul_rmul _ _ := mul_assoc _ _ _


local notation "L" => AddMonoid.End.mulLeft


local notation "R" => AddMonoid.End.mulRight


@[simp]
theorem commute_lmul_rmul (a : A) : Commute (L a) (R a) :=
  AddMonoidHom.ext fun _ => (IsJordan.lmul_comm_rmul _ _).symm


@[simp]
theorem commute_lmul_lmul_sq (a : A) : Commute (L a) (L (a * a)) :=
  AddMonoidHom.ext fun _ => (IsJordan.lmul_lmul_comm_lmul _ _).symm


@[simp]
theorem commute_lmul_rmul_sq (a : A) : Commute (L a) (R (a * a)) :=
  AddMonoidHom.ext fun _ => (IsJordan.lmul_comm_rmul_rmul _ _).symm


@[simp]
theorem commute_lmul_sq_rmul (a : A) : Commute (L (a * a)) (R a) :=
  AddMonoidHom.ext fun _ => IsJordan.lmul_lmul_comm_rmul _ _


@[simp]
theorem commute_rmul_rmul_sq (a : A) : Commute (R a) (R (a * a)) :=
  AddMonoidHom.ext fun _ => (IsJordan.rmul_comm_rmul_rmul _ _).symm


theorem two_nsmul_lie_lmul_lmul_add_eq_lie_lmul_lmul_add [IsCommJordan A] (a b : A) :
    2 • (⁅L a, L (a * b)⁆ + ⁅L b, L (b * a)⁆) = ⁅L (a * a), L b⁆ + ⁅L (b * b), L a⁆ := by
  suffices 2 • ⁅L a, L (a * b)⁆ + 2 • ⁅L b, L (b * a)⁆ + ⁅L b, L (a * a)⁆ + ⁅L a, L (b * b)⁆ = 0 by
    rwa [← sub_eq_zero, ← sub_sub, sub_eq_add_neg, sub_eq_add_neg, lie_skew, lie_skew, nsmul_add]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul 2 (Bracket.bracket (AddMono …
  -/
  convert (commute_lmul_lmul_sq (a + b)).lie_eq using 1
  simp only [add_mul, mul_add, map_add, lie_add, add_lie, mul_comm b a,
    (commute_lmul_lmul_sq a).lie_eq, (commute_lmul_lmul_sq b).lie_eq, zero_add, add_zero, two_smul]
  /-
    case h.e'_2
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/

-- Porting note: the monolithic `calc`-based proof of `two_nsmul_lie_lmul_lmul_add_add_eq_zero`
-- has had four auxiliary parts `aux{0,1,2,3}` split off from it.

private theorem aux0 {a b c : A} : ⁅L (a + b + c), L ((a + b + c) * (a + b + c))⁆ =
    ⁅L a + L b + L c, L (a * a) + L (b * b) + L (c * c) +
    2 • L (a * b) + 2 • L (c * a) + 2 • L (b * c)⁆ := by
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (AddMonoid.End.mulLeft (HAdd.hAdd (HAdd.hAdd a b) c)) (A …
  -/
  rw [add_mul, add_mul]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (AddMonoid.End.mulLeft (HAdd.hAdd (HAdd.hAdd a b) c)) (A …
  -/
  iterate 6 rw [mul_add]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (AddMonoid.End.mulLeft (HAdd.hAdd (HAdd.hAdd a b) c)) (A …
  -/
  iterate 10 rw [map_add]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (HAdd.hAdd (HAdd.hAdd (AddMonoid.End.mulLeft a) (AddMono …
  -/
  rw [mul_comm b a, mul_comm c a, mul_comm c b]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (HAdd.hAdd (HAdd.hAdd (AddMonoid.End.mulLeft a) (AddMono …
  -/
  iterate 3 rw [two_smul]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (HAdd.hAdd (HAdd.hAdd (AddMonoid.End.mulLeft a) (AddMono …
  -/
  simp only [lie_add, add_lie, commute_lmul_lmul_sq, zero_add, add_zero]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracke …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


private theorem aux1 {a b c : A} :
    ⁅L a + L b + L c, L (a * a) + L (b * b) + L (c * c) +
    2 • L (a * b) + 2 • L (c * a) + 2 • L (b * c)⁆
    =
    ⁅L a, L (a * a)⁆ + ⁅L a, L (b * b)⁆ + ⁅L a, L (c * c)⁆ +
    ⁅L a, 2 • L (a * b)⁆ + ⁅L a, 2 • L (c * a)⁆ + ⁅L a, 2 • L (b * c)⁆ +
    (⁅L b, L (a * a)⁆ + ⁅L b, L (b * b)⁆ + ⁅L b, L (c * c)⁆ +
    ⁅L b, 2 • L (a * b)⁆ + ⁅L b, 2 • L (c * a)⁆ + ⁅L b, 2 • L (b * c)⁆) +
    (⁅L c, L (a * a)⁆ + ⁅L c, L (b * b)⁆ + ⁅L c, L (c * c)⁆ +
    ⁅L c, 2 • L (a * b)⁆ + ⁅L c, 2 • L (c * a)⁆ + ⁅L c, 2 • L (b * c)⁆) := by
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (Bracket.bracket (HAdd.hAdd (HAdd.hAdd (AddMonoid.End.mulLeft a) (AddMono …
  -/
  rw [add_lie, add_lie]
  /-
    A : Type u_1
    inst✝ : NonUnitalNonAssocCommRing A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.End.mulLeft a) (HAdd.hA …
  -/
  iterate 15 rw [lie_add]
  /-
    🎉 no goals
  -/


private theorem aux2 {a b c : A} :
    ⁅L a, L (a * a)⁆ + ⁅L a, L (b * b)⁆ + ⁅L a, L (c * c)⁆ +
    ⁅L a, 2 • L (a * b)⁆ + ⁅L a, 2 • L (c * a)⁆ + ⁅L a, 2 • L (b * c)⁆ +
    (⁅L b, L (a * a)⁆ + ⁅L b, L (b * b)⁆ + ⁅L b, L (c * c)⁆ +
    ⁅L b, 2 • L (a * b)⁆ + ⁅L b, 2 • L (c * a)⁆ + ⁅L b, 2 • L (b * c)⁆) +
    (⁅L c, L (a * a)⁆ + ⁅L c, L (b * b)⁆ + ⁅L c, L (c * c)⁆ +
    ⁅L c, 2 • L (a * b)⁆ + ⁅L c, 2 • L (c * a)⁆ + ⁅L c, 2 • L (b * c)⁆)
    =
    ⁅L a, L (b * b)⁆ + ⁅L b, L (a * a)⁆ + 2 • (⁅L a, L (a * b)⁆ + ⁅L b, L (a * b)⁆) +
    (⁅L a, L (c * c)⁆ + ⁅L c, L (a * a)⁆ + 2 • (⁅L a, L (c * a)⁆ + ⁅L c, L (c * a)⁆)) +
    (⁅L b, L (c * c)⁆ + ⁅L c, L (b * b)⁆ + 2 • (⁅L b, L (b * c)⁆ + ⁅L c, L (b * c)⁆)) +
    (2 • ⁅L a, L (b * c)⁆ + 2 • ⁅L b, L (c * a)⁆ + 2 • ⁅L c, L (a * b)⁆) := by
  rw [(commute_lmul_lmul_sq a).lie_eq, (commute_lmul_lmul_sq b).lie_eq,
    (commute_lmul_lmul_sq c).lie_eq, zero_add, add_zero, add_zero]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracke …
  -/
  simp only [lie_nsmul]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracke …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


private theorem aux3 {a b c : A} :
    ⁅L a, L (b * b)⁆ + ⁅L b, L (a * a)⁆ + 2 • (⁅L a, L (a * b)⁆ + ⁅L b, L (a * b)⁆) +
    (⁅L a, L (c * c)⁆ + ⁅L c, L (a * a)⁆ + 2 • (⁅L a, L (c * a)⁆ + ⁅L c, L (c * a)⁆)) +
    (⁅L b, L (c * c)⁆ + ⁅L c, L (b * b)⁆ + 2 • (⁅L b, L (b * c)⁆ + ⁅L c, L (b * c)⁆)) +
    (2 • ⁅L a, L (b * c)⁆ + 2 • ⁅L b, L (c * a)⁆ + 2 • ⁅L c, L (a * b)⁆)
    =
    2 • ⁅L a, L (b * c)⁆ + 2 • ⁅L b, L (c * a)⁆ + 2 • ⁅L c, L (a * b)⁆ := by
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket ( …
  -/
  rw [add_left_eq_self]
  -- Porting note: was `nth_rw` instead of `conv_lhs`
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  conv_lhs => enter [1, 1, 2, 2, 2]; rw [mul_comm a b]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  conv_lhs => enter [1, 2, 2, 2, 1]; rw [mul_comm c a]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  conv_lhs => enter [   2, 2, 2, 2]; rw [mul_comm b c]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  iterate 3 rw [two_nsmul_lie_lmul_lmul_add_eq_lie_lmul_lmul_add]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  iterate 2 rw [← lie_skew (L (a * a)), ← lie_skew (L (b * b)), ← lie_skew (L (c * c))]
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.E …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem two_nsmul_lie_lmul_lmul_add_add_eq_zero (a b c : A) :
    2 • (⁅L a, L (b * c)⁆ + ⁅L b, L (c * a)⁆ + ⁅L c, L (a * b)⁆) = 0 := by
  /-
    A : Type u_1
    inst✝¹ : NonUnitalNonAssocCommRing A
    inst✝ : IsCommJordan A
    a b c : A
    ⊢ Eq (HSMul.hSMul 2 (HAdd.hAdd (HAdd.hAdd (Bracket.bracket (AddMonoid.End.mulL …
  -/
  symm
  calc
    0 = ⁅L (a + b + c), L ((a + b + c) * (a + b + c))⁆ := by
      rw [(commute_lmul_lmul_sq (a + b + c)).lie_eq]
    _ = _ := by rw [aux0, aux1, aux2, aux3, nsmul_add, nsmul_add]

