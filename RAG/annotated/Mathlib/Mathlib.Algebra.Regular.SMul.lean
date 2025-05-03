/-- An `M`-regular element is an element `c` such that multiplication on the left by `c` is an
injective map `M → M`. -/
def IsSMulRegular [SMul R M] (c : R) :=
  Function.Injective ((c • ·) : M → M)


theorem IsLeftRegular.isSMulRegular [Mul R] {c : R} (h : IsLeftRegular c) : IsSMulRegular R c :=
  h


/-- Left-regular multiplication on `R` is equivalent to `R`-regularity of `R` itself. -/
theorem isLeftRegular_iff [Mul R] {a : R} : IsLeftRegular a ↔ IsSMulRegular R a :=
  Iff.rfl


theorem IsRightRegular.isSMulRegular [Mul R] {c : R} (h : IsRightRegular c) :
    IsSMulRegular R (MulOpposite.op c) :=
  h


/-- Right-regular multiplication on `R` is equivalent to `Rᵐᵒᵖ`-regularity of `R` itself. -/
theorem isRightRegular_iff [Mul R] {a : R} :
    IsRightRegular a ↔ IsSMulRegular R (MulOpposite.op a) :=
  Iff.rfl


/-- The product of `M`-regular elements is `M`-regular. -/
theorem smul (ra : IsSMulRegular M a) (rs : IsSMulRegular M s) : IsSMulRegular M (a • s) :=
  fun _ _ ab => rs (ra ((smul_assoc _ _ _).symm.trans (ab.trans (smul_assoc _ _ _))))


/-- If an element `b` becomes `M`-regular after multiplying it on the left by an `M`-regular
element, then `b` is `M`-regular. -/
theorem of_smul (a : R) (ab : IsSMulRegular M (a • s)) : IsSMulRegular M s :=
  @Function.Injective.of_comp _ _ _ (fun m : M => a • m) _ fun c d cd => by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    s : S
    inst✝³ : SMul R M
    inst✝² : SMul R S
    inst✝¹ : SMul S M
    inst✝ : IsScalarTower R S M
    a : R
    ab : IsSMulRegular M (HSMul.hSMul a s)
    c d : M
    cd : Eq (Function.comp (fun m => HSMul.hSMul a m) (fun x => HSMul.hSMul s x) c …
    ⊢ Eq c d
  -/
  dsimp only [Function.comp_def] at cd
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    s : S
    inst✝³ : SMul R M
    inst✝² : SMul R S
    inst✝¹ : SMul S M
    inst✝ : IsScalarTower R S M
    a : R
    ab : IsSMulRegular M (HSMul.hSMul a s)
    c d : M
    cd : Eq (HSMul.hSMul a (HSMul.hSMul s c)) (HSMul.hSMul a (HSMul.hSMul s d))
    ⊢ Eq c d
  -/
  rw [← smul_assoc, ← smul_assoc] at cd
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    s : S
    inst✝³ : SMul R M
    inst✝² : SMul R S
    inst✝¹ : SMul S M
    inst✝ : IsScalarTower R S M
    a : R
    ab : IsSMulRegular M (HSMul.hSMul a s)
    c d : M
    cd : Eq (HSMul.hSMul (HSMul.hSMul a s) c) (HSMul.hSMul (HSMul.hSMul a s) d)
    ⊢ Eq c d
  -/
  exact ab cd
  /-
    🎉 no goals
  -/


/-- An element is `M`-regular if and only if multiplying it on the left by an `M`-regular element
is `M`-regular. -/
@[simp]
theorem smul_iff (b : S) (ha : IsSMulRegular M a) : IsSMulRegular M (a • b) ↔ IsSMulRegular M b :=
  ⟨of_smul _, ha.smul⟩


theorem isLeftRegular [Mul R] {a : R} (h : IsSMulRegular R a) : IsLeftRegular a :=
  h


theorem isRightRegular [Mul R] {a : R} (h : IsSMulRegular R (MulOpposite.op a)) :
    IsRightRegular a :=
  h


theorem mul [Mul R] [IsScalarTower R R M] (ra : IsSMulRegular M a) (rb : IsSMulRegular M b) :
    IsSMulRegular M (a * b) :=
  ra.smul rb


theorem of_mul [Mul R] [IsScalarTower R R M] (ab : IsSMulRegular M (a * b)) :
    IsSMulRegular M b := by
  /-
    R : Type u_1
    M : Type u_3
    a b : R
    inst✝² : SMul R M
    inst✝¹ : Mul R
    inst✝ : IsScalarTower R R M
    ab : IsSMulRegular M (HMul.hMul a b)
    ⊢ IsSMulRegular M b
  -/
  rw [← smul_eq_mul] at ab
  /-
    R : Type u_1
    M : Type u_3
    a b : R
    inst✝² : SMul R M
    inst✝¹ : Mul R
    inst✝ : IsScalarTower R R M
    ab : IsSMulRegular M (HSMul.hSMul a b)
    ⊢ IsSMulRegular M b
  -/
  exact ab.of_smul _
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_iff_right [Mul R] [IsScalarTower R R M] (ha : IsSMulRegular M a) :
    IsSMulRegular M (a * b) ↔ IsSMulRegular M b :=
  ⟨of_mul, ha.mul⟩


/-- Two elements `a` and `b` are `M`-regular if and only if both products `a * b` and `b * a`
are `M`-regular. -/
theorem mul_and_mul_iff [Mul R] [IsScalarTower R R M] :
    IsSMulRegular M (a * b) ∧ IsSMulRegular M (b * a) ↔ IsSMulRegular M a ∧ IsSMulRegular M b := by
  /-
    R : Type u_1
    M : Type u_3
    a b : R
    inst✝² : SMul R M
    inst✝¹ : Mul R
    inst✝ : IsScalarTower R R M
    ⊢ Iff (And (IsSMulRegular M (HMul.hMul a b)) (IsSMulRegular M (HMul.hMul b a)) …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      M : Type u_3
      a b : R
      inst✝² : SMul R M
      inst✝¹ : Mul R
      inst✝ : IsScalarTower R R M
      ⊢ And (IsSMulRegular M (HMul.hMul a b)) (IsSMulRegular M (HMul.hMul b a)) → An …
    -/
  · rintro ⟨ab, ba⟩
    /-
      case refine_1.intro
      R : Type u_1
      M : Type u_3
      a b : R
      inst✝² : SMul R M
      inst✝¹ : Mul R
      inst✝ : IsScalarTower R R M
      ab : IsSMulRegular M (HMul.hMul a b)
      ba : IsSMulRegular M (HMul.hMul b a)
      ⊢ And (IsSMulRegular M a) (IsSMulRegular M b)
    -/
    exact ⟨ba.of_mul, ab.of_mul⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_3
      a b : R
      inst✝² : SMul R M
      inst✝¹ : Mul R
      inst✝ : IsScalarTower R R M
      ⊢ And (IsSMulRegular M a) (IsSMulRegular M b) → And (IsSMulRegular M (HMul.hMu …
    -/
  · rintro ⟨ha, hb⟩
    /-
      case refine_2.intro
      R : Type u_1
      M : Type u_3
      a b : R
      inst✝² : SMul R M
      inst✝¹ : Mul R
      inst✝ : IsScalarTower R R M
      ha : IsSMulRegular M a
      hb : IsSMulRegular M b
      ⊢ And (IsSMulRegular M (HMul.hMul a b)) (IsSMulRegular M (HMul.hMul b a))
    -/
    exact ⟨ha.mul hb, hb.mul ha⟩
    /-
      🎉 no goals
    -/


lemma of_injective {N F} [SMul R N] [FunLike F M N] [MulActionHomClass F R M N]
    (f : F) {r : R} (h1 : Function.Injective f) (h2 : IsSMulRegular N r) :
    IsSMulRegular M r := fun x y h3 => h1 <| h2 <|
  (map_smulₛₗ f r x).symm.trans ((congrArg f h3).trans (map_smulₛₗ f r y))


/-- One is always `M`-regular. -/
@[simp]
theorem one : IsSMulRegular M (1 : R) := fun a b ab => by
  /-
    R : Type u_1
    M : Type u_3
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    a b : M
    ab : Eq ((fun x => HSMul.hSMul 1 x) a) ((fun x => HSMul.hSMul 1 x) b)
    ⊢ Eq a b
  -/
  dsimp only [Function.comp_def] at ab
  /-
    R : Type u_1
    M : Type u_3
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    a b : M
    ab : Eq (HSMul.hSMul 1 a) (HSMul.hSMul 1 b)
    ⊢ Eq a b
  -/
  rw [one_smul, one_smul] at ab
  /-
    R : Type u_1
    M : Type u_3
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    a b : M
    ab : Eq a b
    ⊢ Eq a b
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- An element of `R` admitting a left inverse is `M`-regular. -/
theorem of_mul_eq_one (h : a * b = 1) : IsSMulRegular M b :=
                      /-
                        R : Type u_1
                        M : Type u_3
                        a b : R
                        inst✝¹ : Monoid R
                        inst✝ : MulAction R M
                        h : Eq (HMul.hMul a b) 1
                        ⊢ IsSMulRegular M (HMul.hMul a b)
                      -/
  of_mul (a := a) (by rw [h]; exact one M)
                              /-
                                🎉 no goals
                              -/


/-- Any power of an `M`-regular element is `M`-regular. -/
theorem pow (n : ℕ) (ra : IsSMulRegular M a) : IsSMulRegular M (a ^ n) := by
  induction n with
  | zero => rw [pow_zero]; simp only [one]
  | succ n hn =>
    rw [pow_succ']
    exact (ra.smul_iff (a ^ n)).mpr hn


/-- An element `a` is `M`-regular if and only if a positive power of `a` is `M`-regular. -/
theorem pow_iff {n : ℕ} (n0 : 0 < n) : IsSMulRegular M (a ^ n) ↔ IsSMulRegular M a := by
  /-
    R : Type u_1
    M : Type u_3
    a : R
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    n : Nat
    n0 : LT.lt 0 n
    ⊢ Iff (IsSMulRegular M (HPow.hPow a n)) (IsSMulRegular M a)
  -/
  refine ⟨?_, pow n⟩
  /-
    R : Type u_1
    M : Type u_3
    a : R
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    n : Nat
    n0 : LT.lt 0 n
    ⊢ IsSMulRegular M (HPow.hPow a n) → IsSMulRegular M a
  -/
  rw [← Nat.succ_pred_eq_of_pos n0, pow_succ, ← smul_eq_mul]
  /-
    R : Type u_1
    M : Type u_3
    a : R
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    n : Nat
    n0 : LT.lt 0 n
    ⊢ IsSMulRegular M (HSMul.hSMul (HPow.hPow a n.pred) a) → IsSMulRegular M a
  -/
  exact of_smul _
  /-
    🎉 no goals
  -/


/-- An element of `S` admitting a left inverse in `R` is `M`-regular. -/
theorem of_smul_eq_one (h : a • s = 1) : IsSMulRegular M s :=
  of_smul a
    (by
      /-
        R : Type u_1
        S : Type u_2
        M : Type u_3
        a : R
        s : S
        inst✝⁴ : Monoid S
        inst✝³ : SMul R M
        inst✝² : SMul R S
        inst✝¹ : MulAction S M
        inst✝ : IsScalarTower R S M
        h : Eq (HSMul.hSMul a s) 1
        ⊢ IsSMulRegular M (HSMul.hSMul a s)
      -/
      rw [h]
      /-
        R : Type u_1
        S : Type u_2
        M : Type u_3
        a : R
        s : S
        inst✝⁴ : Monoid S
        inst✝³ : SMul R M
        inst✝² : SMul R S
        inst✝¹ : MulAction S M
        inst✝ : IsScalarTower R S M
        h : Eq (HSMul.hSMul a s) 1
        ⊢ IsSMulRegular M 1
      -/
      exact one M)
      /-
        🎉 no goals
      -/


/-- The element `0` is `M`-regular if and only if `M` is trivial. -/
protected theorem subsingleton (h : IsSMulRegular M (0 : R)) : Subsingleton M :=
                    /-
                      R : Type u_1
                      M : Type u_3
                      inst✝² : MonoidWithZero R
                      inst✝¹ : Zero M
                      inst✝ : MulActionWithZero R M
                      h : IsSMulRegular M 0
                      a b : M
                      ⊢ Eq ((fun x => HSMul.hSMul 0 x) a) ((fun x => HSMul.hSMul 0 x) b)
                    -/
  ⟨fun a b => h (by dsimp only [Function.comp_def]; repeat' rw [MulActionWithZero.zero_smul])⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The element `0` is `M`-regular if and only if `M` is trivial. -/
theorem zero_iff_subsingleton : IsSMulRegular M (0 : R) ↔ Subsingleton M :=
  ⟨fun h => h.subsingleton, fun H a b _ => @Subsingleton.elim _ H a b⟩


/-- The `0` element is not `M`-regular, on a non-trivial module. -/
theorem not_zero_iff : ¬IsSMulRegular M (0 : R) ↔ Nontrivial M := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : MonoidWithZero R
    inst✝¹ : Zero M
    inst✝ : MulActionWithZero R M
    ⊢ Iff (Not (IsSMulRegular M 0)) (Nontrivial M)
  -/
  rw [nontrivial_iff, not_iff_comm, zero_iff_subsingleton, subsingleton_iff]
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : MonoidWithZero R
    inst✝¹ : Zero M
    inst✝ : MulActionWithZero R M
    ⊢ Iff (Not (Exists fun x => Exists fun y => Ne x y)) (∀ (x y : M), Eq x y)
  -/
  push_neg
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : MonoidWithZero R
    inst✝¹ : Zero M
    inst✝ : MulActionWithZero R M
    ⊢ Iff (∀ (x y : M), Eq x y) (∀ (x y : M), Eq x y)
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


/-- The element `0` is `M`-regular when `M` is trivial. -/
theorem zero [sM : Subsingleton M] : IsSMulRegular M (0 : R) :=
  zero_iff_subsingleton.mpr sM


/-- The `0` element is not `M`-regular, on a non-trivial module. -/
theorem not_zero [nM : Nontrivial M] : ¬IsSMulRegular M (0 : R) :=
  not_zero_iff.mpr nM


/-- A product is `M`-regular if and only if the factors are. -/
theorem mul_iff : IsSMulRegular M (a * b) ↔ IsSMulRegular M a ∧ IsSMulRegular M b := by
  /-
    R : Type u_1
    M : Type u_3
    a b : R
    inst✝² : CommSemigroup R
    inst✝¹ : SMul R M
    inst✝ : IsScalarTower R R M
    ⊢ Iff (IsSMulRegular M (HMul.hMul a b)) (And (IsSMulRegular M a) (IsSMulRegula …
  -/
  rw [← mul_and_mul_iff]
  /-
    R : Type u_1
    M : Type u_3
    a b : R
    inst✝² : CommSemigroup R
    inst✝¹ : SMul R M
    inst✝ : IsScalarTower R R M
    ⊢ Iff (IsSMulRegular M (HMul.hMul a b)) (And (IsSMulRegular M (HMul.hMul a b)) …
  -/
  exact ⟨fun ab => ⟨ab, by rwa [mul_comm]⟩, fun rab => rab.1⟩
  /-
    🎉 no goals
  -/


/-- An element of a group acting on a Type is regular. This relies on the availability
of the inverse given by groups, since there is no `LeftCancelSMul` typeclass. -/
theorem isSMulRegular_of_group [MulAction G R] (g : G) : IsSMulRegular R g := by
  /-
    R : Type u_1
    G : Type u_4
    inst✝¹ : Group G
    inst✝ : MulAction G R
    g : G
    ⊢ IsSMulRegular R g
  -/
  intro x y h
  /-
    R : Type u_1
    G : Type u_4
    inst✝¹ : Group G
    inst✝ : MulAction G R
    g : G
    x y : R
    h : Eq ((fun x => HSMul.hSMul g x) x) ((fun x => HSMul.hSMul g x) y)
    ⊢ Eq x y
  -/
                                            /-
                                              🎉 no goals
                                            -/
  convert congr_arg (g⁻¹ • ·) h using 1 <;> simp [← smul_assoc]
                                            /-
                                              🎉 no goals
                                            -/


/-- Any element in `Rˣ` is `M`-regular. -/
theorem Units.isSMulRegular (a : Rˣ) : IsSMulRegular M (a : R) :=
  IsSMulRegular.of_mul_eq_one a.inv_val


/-- A unit is `M`-regular. -/
theorem IsUnit.isSMulRegular (ua : IsUnit a) : IsSMulRegular M a := by
  /-
    R : Type u_1
    M : Type u_3
    a : R
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    ua : IsUnit a
    ⊢ IsSMulRegular M a
  -/
  rcases ua with ⟨a, rfl⟩
  /-
    case intro
    R : Type u_1
    M : Type u_3
    inst✝¹ : Monoid R
    inst✝ : MulAction R M
    a : Units R
    ⊢ IsSMulRegular M ↑a
  -/
  exact a.isSMulRegular M
  /-
    🎉 no goals
  -/


protected
lemma IsSMulRegular.eq_zero_of_smul_eq_zero [Zero M] [SMulZeroClass R M]
    {r : R} {x : M} (h1 : IsSMulRegular M r) (h2 : r • x = 0) : x = 0 :=
  h1 (h2.trans (smul_zero r).symm)


lemma Equiv.isSMulRegular_congr {R S M M'} [SMul R M] [SMul S M'] {e : M ≃ M'}
    {r : R} {s : S} (h : ∀ x, e (r • x) = s • e x) :
    IsSMulRegular M r ↔ IsSMulRegular M' s :=
  (e.comp_injective _).symm.trans  <|
    (iff_of_eq <| congrArg _ <| funext h).trans <| e.injective_comp _

