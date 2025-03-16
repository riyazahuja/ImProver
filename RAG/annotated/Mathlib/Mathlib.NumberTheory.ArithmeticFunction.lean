/-- An arithmetic function is a function from `ℕ` that maps 0 to 0. In the literature, they are
  often instead defined as functions from `ℕ+`. Multiplication on `ArithmeticFunctions` is by
  Dirichlet convolution. -/
def ArithmeticFunction [Zero R] :=
  ZeroHom ℕ R


instance ArithmeticFunction.zero [Zero R] : Zero (ArithmeticFunction R) :=
  inferInstanceAs (Zero (ZeroHom ℕ R))


instance [Zero R] : Inhabited (ArithmeticFunction R) := inferInstanceAs (Inhabited (ZeroHom ℕ R))


instance : FunLike (ArithmeticFunction R) ℕ R :=
  inferInstanceAs (FunLike (ZeroHom ℕ R) ℕ R)


@[simp]
theorem toFun_eq (f : ArithmeticFunction R) : f.toFun = f := rfl


@[simp]
theorem coe_mk (f : ℕ → R) (hf) : @DFunLike.coe (ArithmeticFunction R) _ _ _
    (ZeroHom.mk f hf) = f := rfl


@[simp]
theorem map_zero {f : ArithmeticFunction R} : f 0 = 0 :=
  ZeroHom.map_zero' f


theorem coe_inj {f g : ArithmeticFunction R} : (f : ℕ → R) = g ↔ f = g :=
  DFunLike.coe_fn_eq


@[simp]
theorem zero_apply {x : ℕ} : (0 : ArithmeticFunction R) x = 0 :=
  ZeroHom.zero_apply x


@[ext]
theorem ext ⦃f g : ArithmeticFunction R⦄ (h : ∀ x, f x = g x) : f = g :=
  ZeroHom.ext h


instance one : One (ArithmeticFunction R) :=
  ⟨⟨fun x => ite (x = 1) 1 0, rfl⟩⟩


theorem one_apply {x : ℕ} : (1 : ArithmeticFunction R) x = ite (x = 1) 1 0 :=
  rfl


@[simp]
theorem one_one : (1 : ArithmeticFunction R) 1 = 1 :=
  rfl


@[simp]
theorem one_apply_ne {x : ℕ} (h : x ≠ 1) : (1 : ArithmeticFunction R) x = 0 :=
  if_neg h


/-- Coerce an arithmetic function with values in `ℕ` to one with values in `R`. We cannot inline
this in `natCoe` because it gets unfolded too much. -/
@[coe]  -- Porting note: added `coe` tag.
def natToArithmeticFunction [AddMonoidWithOne R] :
    (ArithmeticFunction ℕ) → (ArithmeticFunction R) :=
                                /-
                                  R : Type u_1
                                  inst✝ : AddMonoidWithOne R
                                  f : ArithmeticFunction Nat
                                  ⊢ Eq ((fun n => ↑(f n)) 0) 0
                                -/
  fun f => ⟨fun n => ↑(f n), by simp⟩
                                /-
                                  🎉 no goals
                                -/


instance natCoe [AddMonoidWithOne R] : Coe (ArithmeticFunction ℕ) (ArithmeticFunction R) :=
  ⟨natToArithmeticFunction⟩


@[simp]
theorem natCoe_nat (f : ArithmeticFunction ℕ) : natToArithmeticFunction f = f :=
  ext fun _ => cast_id _


@[simp]
theorem natCoe_apply [AddMonoidWithOne R] {f : ArithmeticFunction ℕ} {x : ℕ} :
    (f : ArithmeticFunction R) x = f x :=
  rfl


/-- Coerce an arithmetic function with values in `ℤ` to one with values in `R`. We cannot inline
this in `intCoe` because it gets unfolded too much. -/
@[coe]
def ofInt [AddGroupWithOne R] :
    (ArithmeticFunction ℤ) → (ArithmeticFunction R) :=
                                /-
                                  R : Type u_1
                                  inst✝ : AddGroupWithOne R
                                  f : ArithmeticFunction Int
                                  ⊢ Eq ((fun n => ↑(f n)) 0) 0
                                -/
  fun f => ⟨fun n => ↑(f n), by simp⟩
                                /-
                                  🎉 no goals
                                -/


instance intCoe [AddGroupWithOne R] : Coe (ArithmeticFunction ℤ) (ArithmeticFunction R) :=
  ⟨ofInt⟩


@[simp]
theorem intCoe_int (f : ArithmeticFunction ℤ) : ofInt f = f :=
  ext fun _ => Int.cast_id


@[simp]
theorem intCoe_apply [AddGroupWithOne R] {f : ArithmeticFunction ℤ} {x : ℕ} :
    (f : ArithmeticFunction R) x = f x := rfl


@[simp]
theorem coe_coe [AddGroupWithOne R] {f : ArithmeticFunction ℕ} :
    ((f : ArithmeticFunction ℤ) : ArithmeticFunction R) = (f : ArithmeticFunction R) := by
  /-
    R : Type u_1
    inst✝ : AddGroupWithOne R
    f : ArithmeticFunction Nat
    ⊢ Eq ↑↑f ↑f
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : AddGroupWithOne R
    f : ArithmeticFunction Nat
    x✝ : Nat
    ⊢ Eq (↑↑f x✝) (↑f x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem natCoe_one [AddMonoidWithOne R] :
    ((1 : ArithmeticFunction ℕ) : ArithmeticFunction R) = 1 := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (↑1) 1
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    n : Nat
    ⊢ Eq (↑1 n) (1 n)
  -/
  simp [one_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem intCoe_one [AddGroupWithOne R] : ((1 : ArithmeticFunction ℤ) :
    ArithmeticFunction R) = 1 := by
  /-
    R : Type u_1
    inst✝ : AddGroupWithOne R
    ⊢ Eq (↑1) 1
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : AddGroupWithOne R
    n : Nat
    ⊢ Eq (↑1 n) (1 n)
  -/
  simp [one_apply]
  /-
    🎉 no goals
  -/


instance add : Add (ArithmeticFunction R) :=
                                      /-
                                        R : Type u_1
                                        inst✝ : AddMonoid R
                                        f g : ArithmeticFunction R
                                        ⊢ Eq ((fun n => HAdd.hAdd (f n) (g n)) 0) 0
                                      -/
  ⟨fun f g => ⟨fun n => f n + g n, by simp⟩⟩
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem add_apply {f g : ArithmeticFunction R} {n : ℕ} : (f + g) n = f n + g n :=
  rfl


instance instAddMonoid : AddMonoid (ArithmeticFunction R) :=
  { ArithmeticFunction.zero R,
    ArithmeticFunction.add with
    add_assoc := fun _ _ _ => ext fun _ => add_assoc _ _ _
    zero_add := fun _ => ext fun _ => zero_add _
    add_zero := fun _ => ext fun _ => add_zero _
    nsmul := nsmulRec }


instance instAddMonoidWithOne [AddMonoidWithOne R] : AddMonoidWithOne (ArithmeticFunction R) :=
  { ArithmeticFunction.instAddMonoid,
    ArithmeticFunction.one with
                                                                   /-
                                                                     R : Type u_1
                                                                     inst✝ : AddMonoidWithOne R
                                                                     n : Nat
                                                                     ⊢ Eq ((fun x => ite (Eq x 1) (↑n) 0) 0) 0
                                                                   -/
    natCast := fun n => ⟨fun x => if x = 1 then (n : R) else 0, by simp⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                       /-
                         R : Type u_1
                         inst✝ : AddMonoidWithOne R
                         ⊢ Eq (NatCast.natCast 0) 0
                       -/
    natCast_zero := by ext; simp
                            /-
                              🎉 no goals
                            -/
                                /-
                                  R : Type u_1
                                  inst✝ : AddMonoidWithOne R
                                  n : Nat
                                  ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
                                -/
                                                              /-
                                                                🎉 no goals
                                                              -/
    natCast_succ := fun n => by ext x; by_cases h : x = 1 <;> simp [h] }
                                                              /-
                                                                🎉 no goals
                                                              -/


instance instAddCommMonoid [AddCommMonoid R] : AddCommMonoid (ArithmeticFunction R) :=
  { ArithmeticFunction.instAddMonoid with add_comm := fun _ _ => ext fun _ => add_comm _ _ }


instance [NegZeroClass R] : Neg (ArithmeticFunction R) where
                              /-
                                R : Type u_1
                                inst✝ : NegZeroClass R
                                f : ArithmeticFunction R
                                ⊢ Eq ((fun n => Neg.neg (f n)) 0) 0
                              -/
  neg f := ⟨fun n => -f n, by simp⟩
                              /-
                                🎉 no goals
                              -/


instance [AddGroup R] : AddGroup (ArithmeticFunction R) :=
  { ArithmeticFunction.instAddMonoid with
    neg_add_cancel := fun _ => ext fun _ => neg_add_cancel _
    zsmul := zsmulRec }


instance [AddCommGroup R] : AddCommGroup (ArithmeticFunction R) :=
                                            /-
                                              R : Type u_1
                                              inst✝ : AddCommGroup R
                                              ⊢ AddGroup (ArithmeticFunction R)
                                            -/
  { show AddGroup (ArithmeticFunction R) by infer_instance with
                                            /-
                                              🎉 no goals
                                            -/
    add_comm := fun _ _ ↦ add_comm _ _ }


/-- The Dirichlet convolution of two arithmetic functions `f` and `g` is another arithmetic function
  such that `(f * g) n` is the sum of `f x * g y` over all `(x,y)` such that `x * y = n`. -/
instance : SMul (ArithmeticFunction R) (ArithmeticFunction M) :=
                                                                            /-
                                                                              R : Type u_1
                                                                              M : Type u_2
                                                                              inst✝² : Zero R
                                                                              inst✝¹ : AddCommMonoid M
                                                                              inst✝ : SMul R M
                                                                              f : ArithmeticFunction R
                                                                              g : ArithmeticFunction M
                                                                              ⊢ Eq ((fun n => n.divisorsAntidiagonal.sum fun x => HSMul.hSMul (f x.1) (g x.2 …
                                                                            -/
  ⟨fun f g => ⟨fun n => ∑ x ∈ divisorsAntidiagonal n, f x.fst • g x.snd, by simp⟩⟩
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem smul_apply {f : ArithmeticFunction R} {g : ArithmeticFunction M} {n : ℕ} :
    (f • g) n = ∑ x ∈ divisorsAntidiagonal n, f x.fst • g x.snd :=
  rfl


/-- The Dirichlet convolution of two arithmetic functions `f` and `g` is another arithmetic function
  such that `(f * g) n` is the sum of `f x * g y` over all `(x,y)` such that `x * y = n`. -/
instance [Semiring R] : Mul (ArithmeticFunction R) :=
  ⟨(· • ·)⟩


@[simp]
theorem mul_apply [Semiring R] {f g : ArithmeticFunction R} {n : ℕ} :
    (f * g) n = ∑ x ∈ divisorsAntidiagonal n, f x.fst * g x.snd :=
  rfl


                                                                                              /-
                                                                                                R : Type u_1
                                                                                                inst✝ : Semiring R
                                                                                                f g : ArithmeticFunction R
                                                                                                ⊢ Eq ((HMul.hMul f g) 1) (HMul.hMul (f 1) (g 1))
                                                                                              -/
theorem mul_apply_one [Semiring R] {f g : ArithmeticFunction R} : (f * g) 1 = f 1 * g 1 := by simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[simp, norm_cast]
theorem natCoe_mul [Semiring R] {f g : ArithmeticFunction ℕ} :
    (↑(f * g) : ArithmeticFunction R) = f * g := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : ArithmeticFunction Nat
    ⊢ Eq (↑(HMul.hMul f g)) (HMul.hMul ↑f ↑g)
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f g : ArithmeticFunction Nat
    n : Nat
    ⊢ Eq (↑(HMul.hMul f g) n) ((HMul.hMul ↑f ↑g) n)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem intCoe_mul [Ring R] {f g : ArithmeticFunction ℤ} :
    (↑(f * g) : ArithmeticFunction R) = ↑f * g := by
  /-
    R : Type u_1
    inst✝ : Ring R
    f g : ArithmeticFunction Int
    ⊢ Eq (↑(HMul.hMul f g)) (HMul.hMul ↑f ↑g)
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    f g : ArithmeticFunction Int
    n : Nat
    ⊢ Eq (↑(HMul.hMul f g) n) ((HMul.hMul ↑f ↑g) n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mul_smul' (f g : ArithmeticFunction R) (h : ArithmeticFunction M) :
    (f * g) • h = f • g • h := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : ArithmeticFunction R
    h : ArithmeticFunction M
    ⊢ Eq (HSMul.hSMul (HMul.hMul f g) h) (HSMul.hSMul f (HSMul.hSMul g h))
  -/
  ext n
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : ArithmeticFunction R
    h : ArithmeticFunction M
    n : Nat
    ⊢ Eq ((HSMul.hSMul (HMul.hMul f g) h) n) ((HSMul.hSMul f (HSMul.hSMul g h)) n)
  -/
  simp only [mul_apply, smul_apply, sum_smul, mul_smul, smul_sum, Finset.sum_sigma']
  apply Finset.sum_nbij' (fun ⟨⟨_i, j⟩, ⟨k, l⟩⟩ ↦ ⟨(k, l * j), (l, j)⟩)
                                                       /-
                                                         case h.hi
                                                         R : Type u_1
                                                         M : Type u_2
                                                         inst✝² : Semiring R
                                                         inst✝¹ : AddCommMonoid M
                                                         inst✝ : Module R M
                                                         f g : ArithmeticFunction R
                                                         h : ArithmeticFunction M
                                                         n : Nat
                                                         ⊢ ∀ (a : Sigma fun i => Prod Nat Nat), Membership.mem (n.divisorsAntidiagonal. …
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
                                                       /-
                                                         🎉 no goals
                                                       -/
    (fun ⟨⟨i, _j⟩, ⟨k, l⟩⟩ ↦ ⟨(i * k, l), (i, k)⟩) <;> aesop (add simp mul_assoc)
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem one_smul' (b : ArithmeticFunction M) : (1 : ArithmeticFunction R) • b = b := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    ⊢ Eq (HSMul.hSMul 1 b) b
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    x : Nat
    ⊢ Eq ((HSMul.hSMul 1 b) x) (b x)
  -/
  rw [smul_apply]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    x : Nat
    ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HSMul.hSMul (1 x.1) (b x.2)) (b x)
  -/
  by_cases x0 : x = 0
    /-
      case pos
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : ArithmeticFunction M
      x : Nat
      x0 : Eq x 0
      ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HSMul.hSMul (1 x.1) (b x.2)) (b x)
    -/
  · simp [x0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    x : Nat
    x0 : Not (Eq x 0)
    ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HSMul.hSMul (1 x.1) (b x.2)) (b x)
  -/
  have h : {(1, x)} ⊆ divisorsAntidiagonal x := by simp [x0]
  /-
    case neg
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    x : Nat
    x0 : Not (Eq x 0)
    h : HasSubset.Subset (Singleton.singleton { fst := 1, snd := x }) x.divisorsAn …
    ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HSMul.hSMul (1 x.1) (b x.2)) (b x)
  -/
  rw [← sum_subset h]
    /-
      case neg
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : ArithmeticFunction M
      x : Nat
      x0 : Not (Eq x 0)
      h : HasSubset.Subset (Singleton.singleton { fst := 1, snd := x }) x.divisorsAn …
      ⊢ Eq ((Singleton.singleton { fst := 1, snd := x }).sum fun x => HSMul.hSMul (1 …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    x : Nat
    x0 : Not (Eq x 0)
    h : HasSubset.Subset (Singleton.singleton { fst := 1, snd := x }) x.divisorsAn …
    ⊢ ∀ (x_1 : Prod Nat Nat), Membership.mem x.divisorsAntidiagonal x_1 → Not (Mem …
  -/
  intro y ymem ynmem
  have y1ne : y.fst ≠ 1 := by
    intro con
    simp only [mem_divisorsAntidiagonal, one_mul, Ne] at ymem
    simp only [mem_singleton, Prod.ext_iff] at ynmem
    -- Porting note: `tauto` worked from here.
    cases y
    subst con
    simp only [true_and, one_mul, x0, not_false_eq_true, and_true] at ynmem ymem
    tauto

  /-
    case neg
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : ArithmeticFunction M
    x : Nat
    x0 : Not (Eq x 0)
    h : HasSubset.Subset (Singleton.singleton { fst := 1, snd := x }) x.divisorsAn …
    y : Prod Nat Nat
    ymem : Membership.mem x.divisorsAntidiagonal y
    ynmem : Not (Membership.mem (Singleton.singleton { fst := 1, snd := x }) y)
    y1ne : Ne y.1 1
    ⊢ Eq (HSMul.hSMul (1 y.1) (b y.2)) 0
  -/
  simp [y1ne]
  /-
    🎉 no goals
  -/


instance instMonoid : Monoid (ArithmeticFunction R) :=
  { one := One.one
    mul := Mul.mul
    one_mul := one_smul'
    mul_one := fun f => by
      /-
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        ⊢ Eq (HMul.hMul f 1) f
      -/
      ext x
      /-
        case h
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x : Nat
        ⊢ Eq ((HMul.hMul f 1) x) (f x)
      -/
      rw [mul_apply]
      /-
        case h
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x : Nat
        ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (1 x.2)) (f x)
      -/
      by_cases x0 : x = 0
        /-
          case pos
          R : Type u_1
          inst✝ : Semiring R
          f : ArithmeticFunction R
          x : Nat
          x0 : Eq x 0
          ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (1 x.2)) (f x)
        -/
      · simp [x0]
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x : Nat
        x0 : Not (Eq x 0)
        ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (1 x.2)) (f x)
      -/
      have h : {(x, 1)} ⊆ divisorsAntidiagonal x := by simp [x0]
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x : Nat
        x0 : Not (Eq x 0)
        h : HasSubset.Subset (Singleton.singleton { fst := x, snd := 1 }) x.divisorsAn …
        ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (1 x.2)) (f x)
      -/
      rw [← sum_subset h]
        /-
          case neg
          R : Type u_1
          inst✝ : Semiring R
          f : ArithmeticFunction R
          x : Nat
          x0 : Not (Eq x 0)
          h : HasSubset.Subset (Singleton.singleton { fst := x, snd := 1 }) x.divisorsAn …
          ⊢ Eq ((Singleton.singleton { fst := x, snd := 1 }).sum fun x => HMul.hMul (f x …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x : Nat
        x0 : Not (Eq x 0)
        h : HasSubset.Subset (Singleton.singleton { fst := x, snd := 1 }) x.divisorsAn …
        ⊢ ∀ (x_1 : Prod Nat Nat), Membership.mem x.divisorsAntidiagonal x_1 → Not (Mem …
      -/
      intro y ymem ynmem
      have y2ne : y.snd ≠ 1 := by
        intro con
        cases y; subst con -- Porting note: added
        simp only [mem_divisorsAntidiagonal, mul_one, Ne] at ymem
        simp only [mem_singleton, Prod.ext_iff] at ynmem
        tauto
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x : Nat
        x0 : Not (Eq x 0)
        h : HasSubset.Subset (Singleton.singleton { fst := x, snd := 1 }) x.divisorsAn …
        y : Prod Nat Nat
        ymem : Membership.mem x.divisorsAntidiagonal y
        ynmem : Not (Membership.mem (Singleton.singleton { fst := x, snd := 1 }) y)
        y2ne : Ne y.2 1
        ⊢ Eq (HMul.hMul (f y.1) (1 y.2)) 0
      -/
      simp [y2ne]
      /-
        🎉 no goals
      -/
    mul_assoc := mul_smul' }


instance instSemiring : Semiring (ArithmeticFunction R) :=
  -- Porting note: I reorganized this instance
  { ArithmeticFunction.instAddMonoidWithOne,
    ArithmeticFunction.instMonoid,
    ArithmeticFunction.instAddCommMonoid with
    zero_mul := fun f => by
      /-
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        ⊢ Eq (HMul.hMul 0 f) 0
      -/
      ext
      /-
        case h
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x✝ : Nat
        ⊢ Eq ((HMul.hMul 0 f) x✝) (0 x✝)
      -/
      simp only [mul_apply, zero_mul, sum_const_zero, zero_apply]
      /-
        🎉 no goals
      -/
    mul_zero := fun f => by
      /-
        R : Type u_1
        inst✝ : Semiring R
        a b c : ArithmeticFunction R
        ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
      -/
      /-
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        ⊢ Eq (HMul.hMul f 0) 0
      -/
      /-
        case h
        R : Type u_1
        inst✝ : Semiring R
        a b c : ArithmeticFunction R
        x✝ : Nat
        ⊢ Eq ((HMul.hMul a (HAdd.hAdd b c)) x✝) ((HAdd.hAdd (HMul.hMul a b) (HMul.hMul …
      -/
      ext
      /-
        🎉 no goals
      -/
      /-
        case h
        R : Type u_1
        inst✝ : Semiring R
        f : ArithmeticFunction R
        x✝ : Nat
        ⊢ Eq ((HMul.hMul f 0) x✝) (0 x✝)
      -/
      /-
        R : Type u_1
        inst✝ : Semiring R
        a b c : ArithmeticFunction R
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
      -/
      simp only [mul_apply, sum_const_zero, mul_zero, zero_apply]
      /-
        case h
        R : Type u_1
        inst✝ : Semiring R
        a b c : ArithmeticFunction R
        x✝ : Nat
        ⊢ Eq ((HMul.hMul (HAdd.hAdd a b) c) x✝) ((HAdd.hAdd (HMul.hMul a c) (HMul.hMul …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    left_distrib := fun a b c => by
      ext
      simp only [← sum_add_distrib, mul_add, mul_apply, add_apply]
    right_distrib := fun a b c => by
      ext
      simp only [← sum_add_distrib, add_mul, mul_apply, add_apply] }


instance [CommSemiring R] : CommSemiring (ArithmeticFunction R) :=
  { ArithmeticFunction.instSemiring with
    mul_comm := fun f g => by
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        ⊢ Eq (HMul.hMul f g) (HMul.hMul g f)
      -/
      ext
      /-
        case h
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        x✝ : Nat
        ⊢ Eq ((HMul.hMul f g) x✝) ((HMul.hMul g f) x✝)
      -/
      rw [mul_apply, ← map_swap_divisorsAntidiagonal, sum_map]
      /-
        case h
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        x✝ : Nat
        ⊢ Eq (x✝.divisorsAntidiagonal.sum fun x => HMul.hMul (f ((Equiv.prodComm Nat N …
      -/
      simp [mul_comm] }
      /-
        🎉 no goals
      -/


instance [CommRing R] : CommRing (ArithmeticFunction R) :=
  { ArithmeticFunction.instSemiring with
    neg_add_cancel := neg_add_cancel
    mul_comm := mul_comm
    zsmul := (· • ·) }


instance {M : Type*} [Semiring R] [AddCommMonoid M] [Module R M] :
    Module (ArithmeticFunction R) (ArithmeticFunction M) where
  one_smul := one_smul'
  mul_smul := mul_smul'
  smul_add r x y := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : ArithmeticFunction R
      x y : ArithmeticFunction M
      ⊢ Eq (HSMul.hSMul r (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
    -/
    ext
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : ArithmeticFunction R
      x y : ArithmeticFunction M
      x✝ : Nat
      ⊢ Eq ((HSMul.hSMul r (HAdd.hAdd x y)) x✝) ((HAdd.hAdd (HSMul.hSMul r x) (HSMul …
    -/
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : ArithmeticFunction R
      ⊢ Eq (HSMul.hSMul r 0) 0
    -/
    simp only [sum_add_distrib, smul_add, smul_apply, add_apply]
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : ArithmeticFunction R
      x✝ : Nat
      ⊢ Eq ((HSMul.hSMul r 0) x✝) (0 x✝)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  smul_zero r := by
    ext
    simp only [smul_apply, sum_const_zero, smul_zero, zero_apply]
  add_smul r s x := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r s : ArithmeticFunction R
      x : ArithmeticFunction M
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
    -/
    ext
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r s : ArithmeticFunction R
      x : ArithmeticFunction M
      x✝ : Nat
      ⊢ Eq ((HSMul.hSMul (HAdd.hAdd r s) x) x✝) ((HAdd.hAdd (HSMul.hSMul r x) (HSMul …
    -/
    simp only [add_smul, sum_add_distrib, smul_apply, add_apply]
    /-
      🎉 no goals
    -/
  zero_smul r := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : ArithmeticFunction M
      ⊢ Eq (HSMul.hSMul 0 r) 0
    -/
    ext
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : ArithmeticFunction M
      x✝ : Nat
      ⊢ Eq ((HSMul.hSMul 0 r) x✝) (0 x✝)
    -/
    simp only [smul_apply, sum_const_zero, zero_smul, zero_apply]
    /-
      🎉 no goals
    -/


/-- `ζ 0 = 0`, otherwise `ζ x = 1`. The Dirichlet Series is the Riemann `ζ`. -/
def zeta : ArithmeticFunction ℕ :=
  ⟨fun x => ite (x = 0) 0 1, rfl⟩


@[inherit_doc]
scoped[ArithmeticFunction] notation "ζ" => ArithmeticFunction.zeta


@[inherit_doc]
scoped[ArithmeticFunction.zeta] notation "ζ" => ArithmeticFunction.zeta


@[simp]
theorem zeta_apply {x : ℕ} : ζ x = if x = 0 then 0 else 1 :=
  rfl


theorem zeta_apply_ne {x : ℕ} (h : x ≠ 0) : ζ x = 1 :=
  if_neg h

-- Porting note: removed `@[simp]`, LHS not in normal form

theorem coe_zeta_smul_apply {M} [Semiring R] [AddCommMonoid M] [Module R M]
    {f : ArithmeticFunction M} {x : ℕ} :
    ((↑ζ : ArithmeticFunction R) • f) x = ∑ i ∈ divisors x, f i := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : ArithmeticFunction M
    x : Nat
    ⊢ Eq ((HSMul.hSMul (↑ArithmeticFunction.zeta) f) x) (x.divisors.sum fun i => f …
  -/
  rw [smul_apply]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : ArithmeticFunction M
    x : Nat
    ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HSMul.hSMul (↑ArithmeticFunction.zet …
  -/
  trans ∑ i ∈ divisorsAntidiagonal x, f i.snd
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f : ArithmeticFunction M
      x : Nat
      ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HSMul.hSMul (↑ArithmeticFunction.zet …
    -/
  · refine sum_congr rfl fun i hi => ?_
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f : ArithmeticFunction M
      x : Nat
      i : Prod Nat Nat
      hi : Membership.mem x.divisorsAntidiagonal i
      ⊢ Eq (HSMul.hSMul (↑ArithmeticFunction.zeta i.1) (f i.2)) (f i.2)
    -/
    rcases mem_divisorsAntidiagonal.1 hi with ⟨rfl, h⟩
    /-
      case intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f : ArithmeticFunction M
      i : Prod Nat Nat
      hi : Membership.mem (HMul.hMul i.1 i.2).divisorsAntidiagonal i
      h : Ne (HMul.hMul i.1 i.2) 0
      ⊢ Eq (HSMul.hSMul (↑ArithmeticFunction.zeta i.1) (f i.2)) (f i.2)
    -/
    rw [natCoe_apply, zeta_apply_ne (left_ne_zero_of_mul h), cast_one, one_smul]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f : ArithmeticFunction M
      x : Nat
      ⊢ Eq (x.divisorsAntidiagonal.sum fun i => f i.2) (x.divisors.sum fun i => f i)
    -/
  · rw [← map_div_left_divisors, sum_map, Function.Embedding.coeFn_mk]
    /-
      🎉 no goals
    -/

-- Porting note: removed `@[simp]` to make the linter happy.

theorem coe_zeta_mul_apply [Semiring R] {f : ArithmeticFunction R} {x : ℕ} :
    (↑ζ * f) x = ∑ i ∈ divisors x, f i :=
  coe_zeta_smul_apply

-- Porting note: removed `@[simp]` to make the linter happy.

theorem coe_mul_zeta_apply [Semiring R] {f : ArithmeticFunction R} {x : ℕ} :
    (f * ζ) x = ∑ i ∈ divisors x, f i := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    x : Nat
    ⊢ Eq ((HMul.hMul f ↑ArithmeticFunction.zeta) x) (x.divisors.sum fun i => f i)
  -/
  rw [mul_apply]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    x : Nat
    ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (↑ArithmeticFuncti …
  -/
  trans ∑ i ∈ divisorsAntidiagonal x, f i.1
    /-
      R : Type u_1
      inst✝ : Semiring R
      f : ArithmeticFunction R
      x : Nat
      ⊢ Eq (x.divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (↑ArithmeticFuncti …
    -/
  · refine sum_congr rfl fun i hi => ?_
    /-
      R : Type u_1
      inst✝ : Semiring R
      f : ArithmeticFunction R
      x : Nat
      i : Prod Nat Nat
      hi : Membership.mem x.divisorsAntidiagonal i
      ⊢ Eq (HMul.hMul (f i.1) (↑ArithmeticFunction.zeta i.2)) (f i.1)
    -/
    rcases mem_divisorsAntidiagonal.1 hi with ⟨rfl, h⟩
    /-
      case intro
      R : Type u_1
      inst✝ : Semiring R
      f : ArithmeticFunction R
      i : Prod Nat Nat
      hi : Membership.mem (HMul.hMul i.1 i.2).divisorsAntidiagonal i
      h : Ne (HMul.hMul i.1 i.2) 0
      ⊢ Eq (HMul.hMul (f i.1) (↑ArithmeticFunction.zeta i.2)) (f i.1)
    -/
    rw [natCoe_apply, zeta_apply_ne (right_ne_zero_of_mul h), cast_one, mul_one]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : Semiring R
      f : ArithmeticFunction R
      x : Nat
      ⊢ Eq (x.divisorsAntidiagonal.sum fun i => f i.1) (x.divisors.sum fun i => f i)
    -/
  · rw [← map_div_right_divisors, sum_map, Function.Embedding.coeFn_mk]
    /-
      🎉 no goals
    -/


theorem zeta_mul_apply {f : ArithmeticFunction ℕ} {x : ℕ} : (ζ * f) x = ∑ i ∈ divisors x, f i :=
  coe_zeta_mul_apply
  -- Porting note: was `by rw [← nat_coe_nat ζ, coe_zeta_mul_apply]`.  Is this `theorem` obsolete?


theorem mul_zeta_apply {f : ArithmeticFunction ℕ} {x : ℕ} : (f * ζ) x = ∑ i ∈ divisors x, f i :=
  coe_mul_zeta_apply
  -- Porting note: was `by rw [← natCoe_nat ζ, coe_mul_zeta_apply]`.  Is this `theorem` obsolete=


/-- This is the pointwise product of `ArithmeticFunction`s. -/
def pmul [MulZeroClass R] (f g : ArithmeticFunction R) : ArithmeticFunction R :=
                          /-
                            R : Type u_1
                            inst✝ : MulZeroClass R
                            f g : ArithmeticFunction R
                            ⊢ Eq ((fun x => HMul.hMul (f x) (g x)) 0) 0
                          -/
  ⟨fun x => f x * g x, by simp⟩
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem pmul_apply [MulZeroClass R] {f g : ArithmeticFunction R} {x : ℕ} : f.pmul g x = f x * g x :=
  rfl


theorem pmul_comm [CommMonoidWithZero R] (f g : ArithmeticFunction R) : f.pmul g = g.pmul f := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f g : ArithmeticFunction R
    ⊢ Eq (f.pmul g) (g.pmul f)
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f g : ArithmeticFunction R
    x✝ : Nat
    ⊢ Eq ((f.pmul g) x✝) ((g.pmul f) x✝)
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


lemma pmul_assoc [CommMonoidWithZero R] (f₁ f₂ f₃ : ArithmeticFunction R) :
    pmul (pmul f₁ f₂) f₃ = pmul f₁ (pmul f₂ f₃) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f₁ f₂ f₃ : ArithmeticFunction R
    ⊢ Eq ((f₁.pmul f₂).pmul f₃) (f₁.pmul (f₂.pmul f₃))
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f₁ f₂ f₃ : ArithmeticFunction R
    x✝ : Nat
    ⊢ Eq (((f₁.pmul f₂).pmul f₃) x✝) ((f₁.pmul (f₂.pmul f₃)) x✝)
  -/
  simp only [pmul_apply, mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem pmul_zeta (f : ArithmeticFunction R) : f.pmul ↑ζ = f := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    f : ArithmeticFunction R
    ⊢ Eq (f.pmul ↑ArithmeticFunction.zeta) f
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    f : ArithmeticFunction R
    x : Nat
    ⊢ Eq ((f.pmul ↑ArithmeticFunction.zeta) x) (f x)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp [Nat.succ_ne_zero]
              /-
                🎉 no goals
              -/


@[simp]
theorem zeta_pmul (f : ArithmeticFunction R) : (ζ : ArithmeticFunction R).pmul f = f := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    f : ArithmeticFunction R
    ⊢ Eq ((↑ArithmeticFunction.zeta).pmul f) f
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    f : ArithmeticFunction R
    x : Nat
    ⊢ Eq (((↑ArithmeticFunction.zeta).pmul f) x) (f x)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp [Nat.succ_ne_zero]
              /-
                🎉 no goals
              -/


/-- This is the pointwise power of `ArithmeticFunction`s. -/
def ppow (f : ArithmeticFunction R) (k : ℕ) : ArithmeticFunction R :=
                                                 /-
                                                   R : Type u_1
                                                   inst✝ : Semiring R
                                                   f : ArithmeticFunction R
                                                   k : Nat
                                                   h0 : Not (Eq k 0)
                                                   ⊢ Eq ((fun x => HPow.hPow (f x) k) 0) 0
                                                 -/
  if h0 : k = 0 then ζ else ⟨fun x ↦ f x ^ k, by simp_rw [map_zero, zero_pow h0]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                                  /-
                                                                    R : Type u_1
                                                                    inst✝ : Semiring R
                                                                    f : ArithmeticFunction R
                                                                    ⊢ Eq (f.ppow 0) ↑ArithmeticFunction.zeta
                                                                  -/
theorem ppow_zero {f : ArithmeticFunction R} : f.ppow 0 = ζ := by rw [ppow, dif_pos rfl]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem ppow_apply {f : ArithmeticFunction R} {k x : ℕ} (kpos : 0 < k) : f.ppow k x = f x ^ k := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k x : Nat
    kpos : LT.lt 0 k
    ⊢ Eq ((f.ppow k) x) (HPow.hPow (f x) k)
  -/
  rw [ppow, dif_neg (Nat.ne_of_gt kpos)]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k x : Nat
    kpos : LT.lt 0 k
    ⊢ Eq ({ toFun := fun x => HPow.hPow (f x) k, map_zero' := ⋯ } x) (HPow.hPow (f …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ppow_succ' {f : ArithmeticFunction R} {k : ℕ} : f.ppow (k + 1) = f.pmul (f.ppow k) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k : Nat
    ⊢ Eq (f.ppow (HAdd.hAdd k 1)) (f.pmul (f.ppow k))
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k x : Nat
    ⊢ Eq ((f.ppow (HAdd.hAdd k 1)) x) ((f.pmul (f.ppow k)) x)
  -/
  rw [ppow_apply (Nat.succ_pos k), _root_.pow_succ']
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k x : Nat
    ⊢ Eq (HMul.hMul (f x) (HPow.hPow (f x) k)) ((f.pmul (f.ppow k)) x)
  -/
                  /-
                    🎉 no goals
                  -/
  induction k <;> simp
                  /-
                    🎉 no goals
                  -/


theorem ppow_succ {f : ArithmeticFunction R} {k : ℕ} {kpos : 0 < k} :
    f.ppow (k + 1) = (f.ppow k).pmul f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k : Nat
    kpos : LT.lt 0 k
    ⊢ Eq (f.ppow (HAdd.hAdd k 1)) ((f.ppow k).pmul f)
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k : Nat
    kpos : LT.lt 0 k
    x : Nat
    ⊢ Eq ((f.ppow (HAdd.hAdd k 1)) x) (((f.ppow k).pmul f) x)
  -/
  rw [ppow_apply (Nat.succ_pos k), _root_.pow_succ]
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : ArithmeticFunction R
    k : Nat
    kpos : LT.lt 0 k
    x : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (f x) k) (f x)) (((f.ppow k).pmul f) x)
  -/
                  /-
                    🎉 no goals
                  -/
  induction k <;> simp
                  /-
                    🎉 no goals
                  -/


/-- This is the pointwise division of `ArithmeticFunction`s. -/
def pdiv [GroupWithZero R] (f g : ArithmeticFunction R) : ArithmeticFunction R :=
                          /-
                            R : Type u_1
                            inst✝ : GroupWithZero R
                            f g : ArithmeticFunction R
                            ⊢ Eq ((fun n => HDiv.hDiv (f n) (g n)) 0) 0
                          -/
  ⟨fun n => f n / g n, by simp only [map_zero, ne_eq, not_true, div_zero]⟩
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem pdiv_apply [GroupWithZero R] (f g : ArithmeticFunction R) (n : ℕ) :
    pdiv f g n = f n / g n := rfl


/-- This result only holds for `DivisionSemiring`s instead of `GroupWithZero`s because zeta takes
values in ℕ, and hence the coercion requires an `AddMonoidWithOne`. TODO: Generalise zeta -/
@[simp]
theorem pdiv_zeta [DivisionSemiring R] (f : ArithmeticFunction R) :
    pdiv f zeta = f := by
  /-
    R : Type u_1
    inst✝ : DivisionSemiring R
    f : ArithmeticFunction R
    ⊢ Eq (f.pdiv ↑ArithmeticFunction.zeta) f
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : DivisionSemiring R
    f : ArithmeticFunction R
    n : Nat
    ⊢ Eq ((f.pdiv ↑ArithmeticFunction.zeta) n) (f n)
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp [succ_ne_zero]
              /-
                🎉 no goals
              -/


/-- The map $n \mapsto \prod_{p \mid n} f(p)$ as an arithmetic function -/
def prodPrimeFactors [CommMonoidWithZero R] (f : ℕ → R) : ArithmeticFunction R where
  toFun d := if d = 0 then 0 else ∏ p ∈ d.primeFactors, f p
  map_zero' := if_pos rfl


/-- `∏ᵖ p ∣ n, f p` is custom notation for `prodPrimeFactors f n` -/
scoped syntax (name := bigproddvd) "∏ᵖ " extBinder " ∣ " term ", " term:67 : term

scoped macro_rules (kind := bigproddvd)
  | `(∏ᵖ $x:ident ∣ $n, $r) => `(prodPrimeFactors (fun $x ↦ $r) $n)


@[simp]
theorem prodPrimeFactors_apply [CommMonoidWithZero R] {f : ℕ → R} {n : ℕ} (hn : n ≠ 0) :
    ∏ᵖ p ∣ n, f p = ∏ p ∈ n.primeFactors, f p :=
  if_neg hn


/-- Multiplicative functions -/
def IsMultiplicative [MonoidWithZero R] (f : ArithmeticFunction R) : Prop :=
  f 1 = 1 ∧ ∀ {m n : ℕ}, m.Coprime n → f (m * n) = f m * f n


@[simp, arith_mult]
theorem map_one {f : ArithmeticFunction R} (h : f.IsMultiplicative) : f 1 = 1 :=
  h.1


@[simp]
theorem map_mul_of_coprime {f : ArithmeticFunction R} (hf : f.IsMultiplicative) {m n : ℕ}
    (h : m.Coprime n) : f (m * n) = f m * f n :=
  hf.2 h


theorem map_prod {ι : Type*} [CommMonoidWithZero R] (g : ι → ℕ) {f : ArithmeticFunction R}
    (hf : f.IsMultiplicative) (s : Finset ι) (hs : (s : Set ι).Pairwise (Coprime on g)) :
    f (∏ i ∈ s, g i) = ∏ i ∈ s, f (g i) := by
  classical
    induction' s using Finset.induction_on with a s has ih hs
    · simp [hf]
    rw [coe_insert, Set.pairwise_insert_of_symmetric (Coprime.symmetric.comap g)] at hs
    rw [prod_insert has, prod_insert has, hf.map_mul_of_coprime, ih hs.1]
    exact .prod_right fun i hi => hs.2 _ hi (hi.ne_of_not_mem has).symm


theorem map_prod_of_prime [CommSemiring R] {f : ArithmeticFunction R}
    (h_mult : ArithmeticFunction.IsMultiplicative f)
    (t : Finset ℕ) (ht : ∀ p ∈ t, p.Prime) :
    f (∏ a ∈ t, a) = ∏ a ∈ t, f a :=
  map_prod _ h_mult t fun x hx y hy hxy => (coprime_primes (ht x hx) (ht y hy)).mpr hxy


theorem map_prod_of_subset_primeFactors [CommSemiring R] {f : ArithmeticFunction R}
    (h_mult : ArithmeticFunction.IsMultiplicative f) (l : ℕ)
    (t : Finset ℕ) (ht : t ⊆ l.primeFactors) :
    f (∏ a ∈ t, a) = ∏ a ∈ t, f a :=
  map_prod_of_prime h_mult t fun _ a => prime_of_mem_primeFactors (ht a)


theorem map_div_of_coprime [CommGroupWithZero R] {f : ArithmeticFunction R}
    (hf : IsMultiplicative f) {l d : ℕ} (hdl : d ∣ l) (hl : (l/d).Coprime d) (hd : f d ≠ 0) :
    f (l / d) = f l / f d := by
  /-
    R : Type u_1
    inst✝ : CommGroupWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    l d : Nat
    hdl : Dvd.dvd d l
    hl : (HDiv.hDiv l d).Coprime d
    hd : Ne (f d) 0
    ⊢ Eq (f (HDiv.hDiv l d)) (HDiv.hDiv (f l) (f d))
  -/
  apply (div_eq_of_eq_mul hd ..).symm
  /-
    R : Type u_1
    inst✝ : CommGroupWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    l d : Nat
    hdl : Dvd.dvd d l
    hl : (HDiv.hDiv l d).Coprime d
    hd : Ne (f d) 0
    ⊢ Eq (f l) (HMul.hMul (f (HDiv.hDiv l d)) (f d))
  -/
  rw [← hf.right hl, Nat.div_mul_cancel hdl]
  /-
    🎉 no goals
  -/


@[arith_mult]
theorem natCast {f : ArithmeticFunction ℕ} [Semiring R] (h : f.IsMultiplicative) :
    IsMultiplicative (f : ArithmeticFunction R) :=
                                 -- Porting note: was `by simp [cop, h]`
      /-
        R : Type u_1
        f : ArithmeticFunction Nat
        inst✝ : Semiring R
        h : f.IsMultiplicative
        ⊢ Eq (↑f 1) 1
      -/
      /-
        🎉 no goals
      -/
  ⟨by simp [h], fun {m n} cop => by simp [h.2 cop]⟩
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-04-17")]
alias nat_cast := natCast


@[arith_mult]
theorem intCast {f : ArithmeticFunction ℤ} [Ring R] (h : f.IsMultiplicative) :
    IsMultiplicative (f : ArithmeticFunction R) :=
                                 -- Porting note: was `by simp [cop, h]`
      /-
        R : Type u_1
        f : ArithmeticFunction Int
        inst✝ : Ring R
        h : f.IsMultiplicative
        ⊢ Eq (↑f 1) 1
      -/
      /-
        🎉 no goals
      -/
  ⟨by simp [h], fun {m n} cop => by simp [h.2 cop]⟩
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-04-17")]
alias int_cast := intCast


@[arith_mult]
theorem mul [CommSemiring R] {f g : ArithmeticFunction R} (hf : f.IsMultiplicative)
    (hg : g.IsMultiplicative) : IsMultiplicative (f * g) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    ⊢ (HMul.hMul f g).IsMultiplicative
  -/
  refine ⟨by simp [hf.1, hg.1], ?_⟩
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    ⊢ ∀ {m n : Nat}, m.Coprime n → Eq ((HMul.hMul f g) (HMul.hMul m n)) (HMul.hMul …
  -/
  simp only [mul_apply]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    ⊢ ∀ {m n : Nat}, m.Coprime n → Eq ((HMul.hMul m n).divisorsAntidiagonal.sum fu …
  -/
  intro m n cop
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    m n : Nat
    cop : m.Coprime n
    ⊢ Eq ((HMul.hMul m n).divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (g x …
  -/
  rw [sum_mul_sum, ← sum_product']
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    m n : Nat
    cop : m.Coprime n
    ⊢ Eq ((HMul.hMul m n).divisorsAntidiagonal.sum fun x => HMul.hMul (f x.1) (g x …
  -/
  symm
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    m n : Nat
    cop : m.Coprime n
    ⊢ Eq ((SProd.sprod m.divisorsAntidiagonal n.divisorsAntidiagonal).sum fun x => …
  -/
  apply sum_nbij fun ((i, j), k, l) ↦ (i * k, j * l)
    /-
      case hi
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ ∀ (a : Prod (Prod Nat Nat) (Prod Nat Nat)), Membership.mem (SProd.sprod m.di …
    -/
  · rintro ⟨⟨a1, a2⟩, ⟨b1, b2⟩⟩ h
    /-
      case hi.mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      a1 a2 b1 b2 : Nat
      h : Membership.mem (SProd.sprod m.divisorsAntidiagonal n.divisorsAntidiagonal) …
      ⊢ Membership.mem (HMul.hMul m n).divisorsAntidiagonal (ArithmeticFunction.IsMu …
    -/
    simp only [mem_divisorsAntidiagonal, Ne, mem_product] at h
    /-
      case hi.mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      a1 a2 b1 b2 : Nat
      h : And (And (Eq (HMul.hMul a1 a2) m) (Not (Eq m 0))) (And (Eq (HMul.hMul b1 b …
      ⊢ Membership.mem (HMul.hMul m n).divisorsAntidiagonal (ArithmeticFunction.IsMu …
    -/
    rcases h with ⟨⟨rfl, ha⟩, ⟨rfl, hb⟩⟩
    /-
      case hi.mk.mk.mk.intro.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul a1 a2) 0)
      cop : (HMul.hMul a1 a2).Coprime (HMul.hMul b1 b2)
      hb : Not (Eq (HMul.hMul b1 b2) 0)
      ⊢ Membership.mem (HMul.hMul (HMul.hMul a1 a2) (HMul.hMul b1 b2)).divisorsAntid …
    -/
    simp only [mem_divisorsAntidiagonal, Nat.mul_eq_zero, Ne]
    /-
      case hi.mk.mk.mk.intro.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul a1 a2) 0)
      cop : (HMul.hMul a1 a2).Coprime (HMul.hMul b1 b2)
      hb : Not (Eq (HMul.hMul b1 b2) 0)
      ⊢ And (Eq (HMul.hMul (HMul.hMul a1 b1) (HMul.hMul a2 b2)) (HMul.hMul (HMul.hMu …
    -/
    constructor
      /-
        case hi.mk.mk.mk.intro.intro.intro.left
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        a1 a2 b1 b2 : Nat
        ha : Not (Eq (HMul.hMul a1 a2) 0)
        cop : (HMul.hMul a1 a2).Coprime (HMul.hMul b1 b2)
        hb : Not (Eq (HMul.hMul b1 b2) 0)
        ⊢ Eq (HMul.hMul (HMul.hMul a1 b1) (HMul.hMul a2 b2)) (HMul.hMul (HMul.hMul a1  …
      -/
    · ring
      /-
        🎉 no goals
      -/
    /-
      case hi.mk.mk.mk.intro.intro.intro.right
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul a1 a2) 0)
      cop : (HMul.hMul a1 a2).Coprime (HMul.hMul b1 b2)
      hb : Not (Eq (HMul.hMul b1 b2) 0)
      ⊢ Not (Or (Or (Eq a1 0) (Eq a2 0)) (Or (Eq b1 0) (Eq b2 0)))
    -/
    rw [Nat.mul_eq_zero] at *
    /-
      case hi.mk.mk.mk.intro.intro.intro.right
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Or (Eq a1 0) (Eq a2 0))
      cop : (HMul.hMul a1 a2).Coprime (HMul.hMul b1 b2)
      hb : Not (Or (Eq b1 0) (Eq b2 0))
      ⊢ Not (Or (Or (Eq a1 0) (Eq a2 0)) (Or (Eq b1 0) (Eq b2 0)))
    -/
    apply not_or_intro ha hb
    /-
      🎉 no goals
    -/
    /-
      case i_inj
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ Set.InjOn (fun x => ArithmeticFunction.IsMultiplicative.mul.match_1 (fun x = …
    -/
  · simp only [Set.InjOn, mem_coe, mem_divisorsAntidiagonal, Ne, mem_product, Prod.mk.inj_iff]
    /-
      case i_inj
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ ∀ ⦃x₁ : Prod (Prod Nat Nat) (Prod Nat Nat)⦄, And (And (Eq (HMul.hMul x₁.1.1  …
    -/
    rintro ⟨⟨a1, a2⟩, ⟨b1, b2⟩⟩ ⟨⟨rfl, ha⟩, ⟨rfl, hb⟩⟩ ⟨⟨c1, c2⟩, ⟨d1, d2⟩⟩ hcd h
    /-
      case i_inj.mk.mk.mk.intro.intro.intro.mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
      hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      c1 c2 d1 d2 : Nat
      hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
      h : And (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1,  …
      ⊢ Eq { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := b2 } } { fs …
    -/
    simp only [Prod.mk.inj_iff] at h
    /-
      case i_inj.mk.mk.mk.intro.intro.intro.mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
      hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      c1 c2 d1 d2 : Nat
      hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
      h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
      ⊢ Eq { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := b2 } } { fs …
    -/
    ext <;> dsimp only
      /-
        case i_inj.mk.mk.mk.intro.intro.intro.mk.mk.mk.fst.fst
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        a1 a2 b1 b2 : Nat
        ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
        hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        c1 c2 d1 d2 : Nat
        hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
        h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
        ⊢ Eq a1 c1
      -/
    · trans Nat.gcd (a1 * a2) (a1 * b1)
        /-
          R : Type u_1
          inst✝ : CommSemiring R
          f g : ArithmeticFunction R
          hf : f.IsMultiplicative
          hg : g.IsMultiplicative
          a1 a2 b1 b2 : Nat
          ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
          hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          c1 c2 d1 d2 : Nat
          hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
          h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
          ⊢ Eq a1 ((HMul.hMul a1 a2).gcd (HMul.hMul a1 b1))
        -/
      · rw [Nat.gcd_mul_left, cop.coprime_mul_left.coprime_mul_right_right.gcd_eq_one, mul_one]
        /-
          🎉 no goals
        -/
        /-
          R : Type u_1
          inst✝ : CommSemiring R
          f g : ArithmeticFunction R
          hf : f.IsMultiplicative
          hg : g.IsMultiplicative
          a1 a2 b1 b2 : Nat
          ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
          hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          c1 c2 d1 d2 : Nat
          hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
          h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
          ⊢ Eq ((HMul.hMul a1 a2).gcd (HMul.hMul a1 b1)) c1
        -/
      · rw [← hcd.1.1, ← hcd.2.1] at cop
        rw [← hcd.1.1, h.1, Nat.gcd_mul_left,
          cop.coprime_mul_left.coprime_mul_right_right.gcd_eq_one, mul_one]
      /-
        case i_inj.mk.mk.mk.intro.intro.intro.mk.mk.mk.fst.snd
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        a1 a2 b1 b2 : Nat
        ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
        hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        c1 c2 d1 d2 : Nat
        hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
        h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
        ⊢ Eq a2 c2
      -/
    · trans Nat.gcd (a1 * a2) (a2 * b2)
      · rw [mul_comm, Nat.gcd_mul_left, cop.coprime_mul_right.coprime_mul_left_right.gcd_eq_one,
          mul_one]
        /-
          R : Type u_1
          inst✝ : CommSemiring R
          f g : ArithmeticFunction R
          hf : f.IsMultiplicative
          hg : g.IsMultiplicative
          a1 a2 b1 b2 : Nat
          ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
          hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          c1 c2 d1 d2 : Nat
          hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
          h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
          ⊢ Eq ((HMul.hMul a1 a2).gcd (HMul.hMul a2 b2)) c2
        -/
      · rw [← hcd.1.1, ← hcd.2.1] at cop
        rw [← hcd.1.1, h.2, mul_comm, Nat.gcd_mul_left,
          cop.coprime_mul_right.coprime_mul_left_right.gcd_eq_one, mul_one]
      /-
        case i_inj.mk.mk.mk.intro.intro.intro.mk.mk.mk.snd.fst
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        a1 a2 b1 b2 : Nat
        ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
        hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        c1 c2 d1 d2 : Nat
        hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
        h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
        ⊢ Eq b1 d1
      -/
    · trans Nat.gcd (b1 * b2) (a1 * b1)
      · rw [mul_comm, Nat.gcd_mul_right,
          cop.coprime_mul_right.coprime_mul_left_right.symm.gcd_eq_one, one_mul]
        /-
          R : Type u_1
          inst✝ : CommSemiring R
          f g : ArithmeticFunction R
          hf : f.IsMultiplicative
          hg : g.IsMultiplicative
          a1 a2 b1 b2 : Nat
          ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
          hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          c1 c2 d1 d2 : Nat
          hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
          h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
          ⊢ Eq ((HMul.hMul b1 b2).gcd (HMul.hMul a1 b1)) d1
        -/
      · rw [← hcd.1.1, ← hcd.2.1] at cop
        rw [← hcd.2.1, h.1, mul_comm c1 d1, Nat.gcd_mul_left,
          cop.coprime_mul_right.coprime_mul_left_right.symm.gcd_eq_one, mul_one]
      /-
        case i_inj.mk.mk.mk.intro.intro.intro.mk.mk.mk.snd.snd
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        a1 a2 b1 b2 : Nat
        ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
        hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
        c1 c2 d1 d2 : Nat
        hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
        h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
        ⊢ Eq b2 d2
      -/
    · trans Nat.gcd (b1 * b2) (a2 * b2)
      · rw [Nat.gcd_mul_right, cop.coprime_mul_left.coprime_mul_right_right.symm.gcd_eq_one,
          one_mul]
        /-
          R : Type u_1
          inst✝ : CommSemiring R
          f g : ArithmeticFunction R
          hf : f.IsMultiplicative
          hg : g.IsMultiplicative
          a1 a2 b1 b2 : Nat
          ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
          hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
          c1 c2 d1 d2 : Nat
          hcd : And (And (Eq (HMul.hMul { fst := { fst := c1, snd := c2 }, snd := { fst  …
          h : And (Eq (HMul.hMul a1 b1) (HMul.hMul c1 d1)) (Eq (HMul.hMul a2 b2) (HMul.h …
          ⊢ Eq ((HMul.hMul b1 b2).gcd (HMul.hMul a2 b2)) d2
        -/
      · rw [← hcd.1.1, ← hcd.2.1] at cop
        rw [← hcd.2.1, h.2, Nat.gcd_mul_right,
          cop.coprime_mul_left.coprime_mul_right_right.symm.gcd_eq_one, one_mul]
  · simp only [Set.SurjOn, Set.subset_def, mem_coe, mem_divisorsAntidiagonal, Ne, mem_product,
      Set.mem_image, exists_prop, Prod.mk.inj_iff]
    /-
      case i_surj
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ ∀ (x : Prod Nat Nat), And (Eq (HMul.hMul x.1 x.2) (HMul.hMul m n)) (Not (Eq  …
    -/
    rintro ⟨b1, b2⟩ h
    /-
      case i_surj.mk
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      b1 b2 : Nat
      h : And (Eq (HMul.hMul { fst := b1, snd := b2 }.1 { fst := b1, snd := b2 }.2)  …
      ⊢ Exists fun x => And (And (And (Eq (HMul.hMul x.1.1 x.1.2) m) (Not (Eq m 0))) …
    -/
    dsimp at h
    /-
      case i_surj.mk
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      b1 b2 : Nat
      h : And (Eq (HMul.hMul b1 b2) (HMul.hMul m n)) (Not (Eq (HMul.hMul m n) 0))
      ⊢ Exists fun x => And (And (And (Eq (HMul.hMul x.1.1 x.1.2) m) (Not (Eq m 0))) …
    -/
    use ((b1.gcd m, b2.gcd m), (b1.gcd n, b2.gcd n))
    rw [← cop.gcd_mul _, ← cop.gcd_mul _, ← h.1, Nat.gcd_mul_gcd_of_coprime_of_mul_eq_mul cop h.1,
      Nat.gcd_mul_gcd_of_coprime_of_mul_eq_mul cop.symm _]
      /-
        case h
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        m n : Nat
        cop : m.Coprime n
        b1 b2 : Nat
        h : And (Eq (HMul.hMul b1 b2) (HMul.hMul m n)) (Not (Eq (HMul.hMul m n) 0))
        ⊢ And (And (And (Eq m m) (Not (Eq m 0))) (And (Eq n n) (Not (Eq n 0)))) (Eq {  …
      -/
    · rw [Nat.mul_eq_zero, not_or] at h
      /-
        case h
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        m n : Nat
        cop : m.Coprime n
        b1 b2 : Nat
        h : And (Eq (HMul.hMul b1 b2) (HMul.hMul m n)) (And (Not (Eq m 0)) (Not (Eq n  …
        ⊢ And (And (And (Eq m m) (Not (Eq m 0))) (And (Eq n n) (Not (Eq n 0)))) (Eq {  …
      -/
      simp [h.2.1, h.2.2]
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      b1 b2 : Nat
      h : And (Eq (HMul.hMul b1 b2) (HMul.hMul m n)) (Not (Eq (HMul.hMul m n) 0))
      ⊢ Eq (HMul.hMul b1 b2) (HMul.hMul n m)
    -/
    rw [mul_comm n m, h.1]
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ ∀ (a : Prod (Prod Nat Nat) (Prod Nat Nat)), Membership.mem (SProd.sprod m.di …
    -/
  · simp only [mem_divisorsAntidiagonal, Ne, mem_product]
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ ∀ (a : Prod (Prod Nat Nat) (Prod Nat Nat)), And (And (Eq (HMul.hMul a.1.1 a. …
    -/
    rintro ⟨⟨a1, a2⟩, ⟨b1, b2⟩⟩ ⟨⟨rfl, ha⟩, ⟨rfl, hb⟩⟩
    /-
      case h.mk.mk.mk.intro.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
      hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      ⊢ Eq (HMul.hMul (HMul.hMul (f { fst := { fst := a1, snd := a2 }, snd := { fst  …
    -/
    dsimp only
    rw [hf.map_mul_of_coprime cop.coprime_mul_right.coprime_mul_right_right,
      hg.map_mul_of_coprime cop.coprime_mul_left.coprime_mul_left_right]
    /-
      case h.mk.mk.mk.intro.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      a1 a2 b1 b2 : Nat
      ha : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      cop : (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, snd := …
      hb : Not (Eq (HMul.hMul { fst := { fst := a1, snd := a2 }, snd := { fst := b1, …
      ⊢ Eq (HMul.hMul (HMul.hMul (f a1) (g a2)) (HMul.hMul (f b1) (g b2))) (HMul.hMu …
    -/
    ring
    /-
      🎉 no goals
    -/


@[arith_mult]
theorem pmul [CommSemiring R] {f g : ArithmeticFunction R} (hf : f.IsMultiplicative)
    (hg : g.IsMultiplicative) : IsMultiplicative (f.pmul g) :=
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        f g : ArithmeticFunction R
        hf : f.IsMultiplicative
        hg : g.IsMultiplicative
        ⊢ Eq ((f.pmul g) 1) 1
      -/
  ⟨by simp [hf, hg], fun {m n} cop => by
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ Eq ((f.pmul g) (HMul.hMul m n)) (HMul.hMul ((f.pmul g) m) ((f.pmul g) n))
    -/
    simp only [pmul_apply, hf.map_mul_of_coprime cop, hg.map_mul_of_coprime cop]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ Eq (HMul.hMul (HMul.hMul (f m) (f n)) (HMul.hMul (g m) (g n))) (HMul.hMul (H …
    -/
    ring⟩
    /-
      🎉 no goals
    -/


@[arith_mult]
theorem pdiv [CommGroupWithZero R] {f g : ArithmeticFunction R} (hf : IsMultiplicative f)
    (hg : IsMultiplicative g) : IsMultiplicative (pdiv f g) :=
       /-
         R : Type u_1
         inst✝ : CommGroupWithZero R
         f g : ArithmeticFunction R
         hf : f.IsMultiplicative
         hg : g.IsMultiplicative
         ⊢ Eq ((f.pdiv g) 1) 1
       -/
  ⟨ by simp [hf, hg], fun {m n} cop => by
       /-
         🎉 no goals
       -/
    simp only [pdiv_apply, map_mul_of_coprime hf cop, map_mul_of_coprime hg cop,
      div_eq_mul_inv, mul_inv]
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : ArithmeticFunction R
      hf : f.IsMultiplicative
      hg : g.IsMultiplicative
      m n : Nat
      cop : m.Coprime n
      ⊢ Eq (HMul.hMul (HMul.hMul (f m) (f n)) (HMul.hMul (Inv.inv (g m)) (Inv.inv (g …
    -/
    apply mul_mul_mul_comm ⟩
    /-
      🎉 no goals
    -/


/-- For any multiplicative function `f` and any `n > 0`,
we can evaluate `f n` by evaluating `f` at `p ^ k` over the factorization of `n` -/
nonrec  -- Porting note: added
theorem multiplicative_factorization [CommMonoidWithZero R] (f : ArithmeticFunction R)
    (hf : f.IsMultiplicative) {n : ℕ} (hn : n ≠ 0) :
    f n = n.factorization.prod fun p k => f (p ^ k) :=
  multiplicative_factorization f (fun _ _ => hf.2) hf.1 hn


/-- A recapitulation of the definition of multiplicative that is simpler for proofs -/
theorem iff_ne_zero [MonoidWithZero R] {f : ArithmeticFunction R} :
    IsMultiplicative f ↔
      f 1 = 1 ∧ ∀ {m n : ℕ}, m ≠ 0 → n ≠ 0 → m.Coprime n → f (m * n) = f m * f n := by
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    f : ArithmeticFunction R
    ⊢ Iff f.IsMultiplicative (And (Eq (f 1) 1) (∀ {m n : Nat}, Ne m 0 → Ne n 0 → m …
  -/
  refine and_congr_right' (forall₂_congr fun m n => ⟨fun h _ _ => h, fun h hmn => ?_⟩)
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    f : ArithmeticFunction R
    m n : Nat
    h : Ne m 0 → Ne n 0 → m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hmn : m.Coprime n
    ⊢ Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f n))
  -/
  rcases eq_or_ne m 0 with (rfl | hm)
    /-
      case inl
      R : Type u_1
      inst✝ : MonoidWithZero R
      f : ArithmeticFunction R
      n : Nat
      h : Ne 0 0 → Ne n 0 → Nat.Coprime 0 n → Eq (f (HMul.hMul 0 n)) (HMul.hMul (f 0 …
      hmn : Nat.Coprime 0 n
      ⊢ Eq (f (HMul.hMul 0 n)) (HMul.hMul (f 0) (f n))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝ : MonoidWithZero R
    f : ArithmeticFunction R
    m n : Nat
    h : Ne m 0 → Ne n 0 → m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hmn : m.Coprime n
    hm : Ne m 0
    ⊢ Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f n))
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inr.inl
      R : Type u_1
      inst✝ : MonoidWithZero R
      f : ArithmeticFunction R
      m : Nat
      hm : Ne m 0
      h : Ne m 0 → Ne 0 0 → m.Coprime 0 → Eq (f (HMul.hMul m 0)) (HMul.hMul (f m) (f …
      hmn : m.Coprime 0
      ⊢ Eq (f (HMul.hMul m 0)) (HMul.hMul (f m) (f 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    R : Type u_1
    inst✝ : MonoidWithZero R
    f : ArithmeticFunction R
    m n : Nat
    h : Ne m 0 → Ne n 0 → m.Coprime n → Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f …
    hmn : m.Coprime n
    hm : Ne m 0
    hn : Ne n 0
    ⊢ Eq (f (HMul.hMul m n)) (HMul.hMul (f m) (f n))
  -/
  exact h hm hn hmn
  /-
    🎉 no goals
  -/


/-- Two multiplicative functions `f` and `g` are equal if and only if
they agree on prime powers -/
theorem eq_iff_eq_on_prime_powers [CommMonoidWithZero R] (f : ArithmeticFunction R)
    (hf : f.IsMultiplicative) (g : ArithmeticFunction R) (hg : g.IsMultiplicative) :
    f = g ↔ ∀ p i : ℕ, Nat.Prime p → f (p ^ i) = g (p ^ i) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    g : ArithmeticFunction R
    hg : g.IsMultiplicative
    ⊢ Iff (Eq f g) (∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.h …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      g : ArithmeticFunction R
      hg : g.IsMultiplicative
      ⊢ Eq f g → ∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p …
    -/
  · intro h p i _
    /-
      case mp
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      g : ArithmeticFunction R
      hg : g.IsMultiplicative
      h : Eq f g
      p i : Nat
      a✝ : Nat.Prime p
      ⊢ Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))
    -/
    rw [h]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    g : ArithmeticFunction R
    hg : g.IsMultiplicative
    ⊢ (∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))) →  …
  -/
  intro h
  /-
    case mpr
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    g : ArithmeticFunction R
    hg : g.IsMultiplicative
    h : ∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))
    ⊢ Eq f g
  -/
  ext n
  /-
    case mpr.h
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    g : ArithmeticFunction R
    hg : g.IsMultiplicative
    h : ∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))
    n : Nat
    ⊢ Eq (f n) (g n)
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      g : ArithmeticFunction R
      hg : g.IsMultiplicative
      h : ∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))
      n : Nat
      hn : Eq n 0
      ⊢ Eq (f n) (g n)
    -/
  · rw [hn, ArithmeticFunction.map_zero, ArithmeticFunction.map_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    g : ArithmeticFunction R
    hg : g.IsMultiplicative
    h : ∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (f n) (g n)
  -/
  rw [multiplicative_factorization f hf hn, multiplicative_factorization g hg hn]
  /-
    case neg
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    g : ArithmeticFunction R
    hg : g.IsMultiplicative
    h : ∀ (p i : Nat), Nat.Prime p → Eq (f (HPow.hPow p i)) (g (HPow.hPow p i))
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (n.factorization.prod fun p k => f (HPow.hPow p k)) (n.factorization.prod …
  -/
  exact Finset.prod_congr rfl fun p hp ↦ h p _ (Nat.prime_of_mem_primeFactors hp)
  /-
    🎉 no goals
  -/


@[arith_mult]
theorem prodPrimeFactors [CommMonoidWithZero R] (f : ℕ → R) :
    IsMultiplicative (prodPrimeFactors f) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : Nat → R
    ⊢ (ArithmeticFunction.prodPrimeFactors f).IsMultiplicative
  -/
  rw [iff_ne_zero]
  simp only [ne_eq, one_ne_zero, not_false_eq_true, prodPrimeFactors_apply, primeFactors_one,
    prod_empty, true_and]
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : Nat → R
    ⊢ ∀ {m n : Nat}, Not (Eq m 0) → Not (Eq n 0) → m.Coprime n → Eq ((ArithmeticFu …
  -/
  intro x y hx hy hxy
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : Nat → R
    x y : Nat
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    hxy : x.Coprime y
    ⊢ Eq ((ArithmeticFunction.prodPrimeFactors f) (HMul.hMul x y)) (HMul.hMul ((Ar …
  -/
  have hxy₀ : x * y ≠ 0 := mul_ne_zero hx hy
  rw [prodPrimeFactors_apply hxy₀, prodPrimeFactors_apply hx, prodPrimeFactors_apply hy,
    Nat.primeFactors_mul hx hy, ← Finset.prod_union hxy.disjoint_primeFactors]


theorem prodPrimeFactors_add_of_squarefree [CommSemiring R] {f g : ArithmeticFunction R}
    (hf : IsMultiplicative f) (hg : IsMultiplicative g) {n : ℕ} (hn : Squarefree n) :
    ∏ᵖ p ∣ n, (f + g) p = (f * g) n := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    n : Nat
    hn : Squarefree n
    ⊢ Eq ((ArithmeticFunction.prodPrimeFactors fun p => (HAdd.hAdd f g) p) n) ((HM …
  -/
  rw [prodPrimeFactors_apply hn.ne_zero]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    n : Nat
    hn : Squarefree n
    ⊢ Eq (n.primeFactors.prod fun p => (HAdd.hAdd f g) p) ((HMul.hMul f g) n)
  -/
  simp_rw [add_apply (f := f) (g := g)]
  rw [Finset.prod_add, mul_apply, sum_divisorsAntidiagonal (f · * g ·),
    ← divisors_filter_squarefree_of_squarefree hn, sum_divisors_filter_squarefree hn.ne_zero,
    factors_eq]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    n : Nat
    hn : Squarefree n
    ⊢ Eq (n.primeFactors.powerset.sum fun t => HMul.hMul (t.prod fun i => f i) ((S …
  -/
  apply Finset.sum_congr rfl
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f g : ArithmeticFunction R
    hf : f.IsMultiplicative
    hg : g.IsMultiplicative
    n : Nat
    hn : Squarefree n
    ⊢ ∀ (x : Finset Nat), Membership.mem n.primeFactors.powerset x → Eq (HMul.hMul …
  -/
  intro t ht
  rw [t.prod_val, Function.id_def,
    ← prod_primeFactors_sdiff_of_squarefree hn (Finset.mem_powerset.mp ht),
    hf.map_prod_of_subset_primeFactors n t (Finset.mem_powerset.mp ht),
    ← hg.map_prod_of_subset_primeFactors n (_ \ t) Finset.sdiff_subset]


theorem lcm_apply_mul_gcd_apply [CommMonoidWithZero R] {f : ArithmeticFunction R}
    (hf : f.IsMultiplicative) {x y : ℕ} :
    f (x.lcm y) * f (x.gcd y) = f x * f y := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    x y : Nat
    ⊢ Eq (HMul.hMul (f (x.lcm y)) (f (x.gcd y))) (HMul.hMul (f x) (f y))
  -/
  by_cases hx : x = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Eq x 0
      ⊢ Eq (HMul.hMul (f (x.lcm y)) (f (x.gcd y))) (HMul.hMul (f x) (f y))
    -/
  · simp only [hx, f.map_zero, zero_mul, Nat.lcm_zero_left, Nat.gcd_zero_left]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    x y : Nat
    hx : Not (Eq x 0)
    ⊢ Eq (HMul.hMul (f (x.lcm y)) (f (x.gcd y))) (HMul.hMul (f x) (f y))
  -/
  by_cases hy : y = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Eq y 0
      ⊢ Eq (HMul.hMul (f (x.lcm y)) (f (x.gcd y))) (HMul.hMul (f x) (f y))
    -/
  · simp only [hy, f.map_zero, mul_zero, Nat.lcm_zero_right, Nat.gcd_zero_right, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    x y : Nat
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Eq (HMul.hMul (f (x.lcm y)) (f (x.gcd y))) (HMul.hMul (f x) (f y))
  -/
  have hgcd_ne_zero : x.gcd y ≠ 0 := gcd_ne_zero_left hx
  /-
    case neg
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    x y : Nat
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    hgcd_ne_zero : Ne (x.gcd y) 0
    ⊢ Eq (HMul.hMul (f (x.lcm y)) (f (x.gcd y))) (HMul.hMul (f x) (f y))
  -/
  have hlcm_ne_zero : x.lcm y ≠ 0 := lcm_ne_zero hx hy
  have hfi_zero : ∀ {i}, f (i ^ 0) = 1 := by
    intro i; rw [Nat.pow_zero, hf.1]
  iterate 4 rw [hf.multiplicative_factorization f (by assumption),
    Finsupp.prod_of_support_subset _ _ _ (fun _ _ => hfi_zero)
      (s := (x.primeFactors ⊔ y.primeFactors))]
    /-
      case neg
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ Eq (HMul.hMul ((Max.max x.primeFactors y.primeFactors).prod fun x_1 => f (HP …
    -/
  · rw [← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
    /-
      case neg
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ Eq ((Max.max x.primeFactors y.primeFactors).prod fun x_1 => HMul.hMul (f (HP …
    -/
    apply Finset.prod_congr rfl
    /-
      case neg
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ ∀ (x_1 : Nat), Membership.mem (Max.max x.primeFactors y.primeFactors) x_1 →  …
    -/
    intro p _
    /-
      case neg
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      p : Nat
      a✝ : Membership.mem (Max.max x.primeFactors y.primeFactors) p
      ⊢ Eq (HMul.hMul (f (HPow.hPow p ((x.lcm y).factorization p))) (f (HPow.hPow p  …
    -/
    rcases Nat.le_or_le (x.factorization p) (y.factorization p) with h | h <;>
      simp only [factorization_lcm hx hy, Finsupp.sup_apply, h, sup_of_le_right,
        sup_of_le_left, inf_of_le_right, Nat.factorization_gcd hx hy, Finsupp.inf_apply,
        inf_of_le_left, mul_comm]
    /-
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ HasSubset.Subset y.factorization.support (Max.max x.primeFactors y.primeFact …
    -/
  · apply Finset.subset_union_right
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ HasSubset.Subset x.factorization.support (Max.max x.primeFactors y.primeFact …
    -/
  · apply Finset.subset_union_left
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ HasSubset.Subset (x.gcd y).factorization.support (Max.max x.primeFactors y.p …
    -/
  · rw [factorization_gcd hx hy, Finsupp.support_inf, Finset.sup_eq_union]
    /-
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ HasSubset.Subset (Inter.inter x.factorization.support y.factorization.suppor …
    -/
    apply Finset.inter_subset_union
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      x y : Nat
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hgcd_ne_zero : Ne (x.gcd y) 0
      hlcm_ne_zero : Ne (x.lcm y) 0
      hfi_zero : ∀ {i : Nat}, Eq (f (HPow.hPow i 0)) 1
      ⊢ HasSubset.Subset (x.lcm y).factorization.support (Max.max x.primeFactors y.p …
    -/
  · simp [factorization_lcm hx hy]
    /-
      🎉 no goals
    -/


theorem map_gcd [CommGroupWithZero R] {f : ArithmeticFunction R}
    (hf : f.IsMultiplicative) {x y : ℕ} (hf_lcm : f (x.lcm y) ≠ 0) :
    f (x.gcd y) = f x * f y / f (x.lcm y) := by
  /-
    R : Type u_1
    inst✝ : CommGroupWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    x y : Nat
    hf_lcm : Ne (f (x.lcm y)) 0
    ⊢ Eq (f (x.gcd y)) (HDiv.hDiv (HMul.hMul (f x) (f y)) (f (x.lcm y)))
  -/
  rw [←hf.lcm_apply_mul_gcd_apply, mul_div_cancel_left₀ _ hf_lcm]
  /-
    🎉 no goals
  -/


theorem map_lcm [CommGroupWithZero R] {f : ArithmeticFunction R}
    (hf : f.IsMultiplicative) {x y : ℕ} (hf_gcd : f (x.gcd y) ≠ 0) :
    f (x.lcm y) = f x * f y / f (x.gcd y) := by
  /-
    R : Type u_1
    inst✝ : CommGroupWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    x y : Nat
    hf_gcd : Ne (f (x.gcd y)) 0
    ⊢ Eq (f (x.lcm y)) (HDiv.hDiv (HMul.hMul (f x) (f y)) (f (x.gcd y)))
  -/
  rw [←hf.lcm_apply_mul_gcd_apply, mul_div_cancel_right₀ _ hf_gcd]
  /-
    🎉 no goals
  -/


theorem eq_zero_of_squarefree_of_dvd_eq_zero [CommMonoidWithZero R] {f : ArithmeticFunction R}
    (hf : IsMultiplicative f) {m n : ℕ} (hn : Squarefree n) (hmn : m ∣ n)
    (h_zero : f m = 0) :
    f n = 0 := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    m n : Nat
    hn : Squarefree n
    hmn : Dvd.dvd m n
    h_zero : Eq (f m) 0
    ⊢ Eq (f n) 0
  -/
  rcases hmn with ⟨k, rfl⟩
  simp only [MulZeroClass.zero_mul, eq_self_iff_true, hf.map_mul_of_coprime
    (coprime_of_squarefree_mul hn), h_zero]


/-- The identity on `ℕ` as an `ArithmeticFunction`. -/
nonrec  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): added
def id : ArithmeticFunction ℕ :=
  ⟨id, rfl⟩


@[simp]
theorem id_apply {x : ℕ} : id x = x :=
  rfl


/-- `pow k n = n ^ k`, except `pow 0 0 = 0`. -/
def pow (k : ℕ) : ArithmeticFunction ℕ :=
  id.ppow k


@[simp]
theorem pow_apply {k n : ℕ} : pow k n = if k = 0 ∧ n = 0 then 0 else n ^ k := by
  /-
    k n : Nat
    ⊢ Eq ((ArithmeticFunction.pow k) n) (ite (And (Eq k 0) (Eq n 0)) 0 (HPow.hPow  …
  -/
  cases k
    /-
      case zero
      n : Nat
      ⊢ Eq ((ArithmeticFunction.pow 0) n) (ite (And (Eq 0 0) (Eq n 0)) 0 (HPow.hPow  …
    -/
  · simp [pow]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n n✝ : Nat
    ⊢ Eq ((ArithmeticFunction.pow (HAdd.hAdd n✝ 1)) n) (ite (And (Eq (HAdd.hAdd n✝ …
  -/
  rename_i k  -- Porting note: added
  /-
    case succ
    n k : Nat
    ⊢ Eq ((ArithmeticFunction.pow (HAdd.hAdd k 1)) n) (ite (And (Eq (HAdd.hAdd k 1 …
  -/
  simp [pow, k.succ_pos.ne']
  /-
    🎉 no goals
  -/


theorem pow_zero_eq_zeta : pow 0 = ζ := by
  /-
    ⊢ Eq (ArithmeticFunction.pow 0) ArithmeticFunction.zeta
  -/
  ext n
  /-
    case h
    n : Nat
    ⊢ Eq ((ArithmeticFunction.pow 0) n) (ArithmeticFunction.zeta n)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `σ k n` is the sum of the `k`th powers of the divisors of `n` -/
def sigma (k : ℕ) : ArithmeticFunction ℕ :=
                                        /-
                                          R : Type u_1
                                          k : Nat
                                          ⊢ Eq ((fun n => n.divisors.sum fun d => HPow.hPow d k) 0) 0
                                        -/
  ⟨fun n => ∑ d ∈ divisors n, d ^ k, by simp⟩
                                        /-
                                          🎉 no goals
                                        -/


@[inherit_doc]
scoped[ArithmeticFunction] notation "σ" => ArithmeticFunction.sigma


@[inherit_doc]
scoped[ArithmeticFunction.sigma] notation "σ" => ArithmeticFunction.sigma


theorem sigma_apply {k n : ℕ} : σ k n = ∑ d ∈ divisors n, d ^ k :=
  rfl


theorem sigma_apply_prime_pow {k p i : ℕ} (hp : p.Prime) :
    σ k (p ^ i) = ∑ j in .range (i + 1), p ^ (j * k) := by
  /-
    k p i : Nat
    hp : Nat.Prime p
    ⊢ Eq ((ArithmeticFunction.sigma k) (HPow.hPow p i)) ((Finset.range (HAdd.hAdd  …
  -/
  simp [sigma_apply, divisors_prime_pow hp, Nat.pow_mul]
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      n : Nat
                                                                      ⊢ Eq ((ArithmeticFunction.sigma 1) n) (n.divisors.sum fun d => d)
                                                                    -/
theorem sigma_one_apply (n : ℕ) : σ 1 n = ∑ d ∈ divisors n, d := by simp [sigma_apply]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem sigma_one_apply_prime_pow {p i : ℕ} (hp : p.Prime) :
    σ 1 (p ^ i) = ∑ k in .range (i + 1), p ^ k := by
  /-
    p i : Nat
    hp : Nat.Prime p
    ⊢ Eq ((ArithmeticFunction.sigma 1) (HPow.hPow p i)) ((Finset.range (HAdd.hAdd  …
  -/
  simp [sigma_apply_prime_pow hp]
  /-
    🎉 no goals
  -/


                                                             /-
                                                               n : Nat
                                                               ⊢ Eq ((ArithmeticFunction.sigma 0) n) n.divisors.card
                                                             -/
theorem sigma_zero_apply (n : ℕ) : σ 0 n = #n.divisors := by simp [sigma_apply]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem sigma_zero_apply_prime_pow {p i : ℕ} (hp : p.Prime) : σ 0 (p ^ i) = i + 1 := by
  /-
    p i : Nat
    hp : Nat.Prime p
    ⊢ Eq ((ArithmeticFunction.sigma 0) (HPow.hPow p i)) (HAdd.hAdd i 1)
  -/
  simp [sigma_apply_prime_pow hp]
  /-
    🎉 no goals
  -/


theorem zeta_mul_pow_eq_sigma {k : ℕ} : ζ * pow k = σ k := by
  /-
    k : Nat
    ⊢ Eq (HMul.hMul ArithmeticFunction.zeta (ArithmeticFunction.pow k)) (Arithmeti …
  -/
  ext
  /-
    case h
    k x✝ : Nat
    ⊢ Eq ((HMul.hMul ArithmeticFunction.zeta (ArithmeticFunction.pow k)) x✝) ((Ari …
  -/
  rw [sigma, zeta_mul_apply]
  /-
    case h
    k x✝ : Nat
    ⊢ Eq (x✝.divisors.sum fun i => (ArithmeticFunction.pow k) i) ({ toFun := fun n …
  -/
  apply sum_congr rfl
  /-
    case h
    k x✝ : Nat
    ⊢ ∀ (x : Nat), Membership.mem x✝.divisors x → Eq ((ArithmeticFunction.pow k) x …
  -/
  intro x hx
  /-
    case h
    k x✝ x : Nat
    hx : Membership.mem x✝.divisors x
    ⊢ Eq ((ArithmeticFunction.pow k) x) (HPow.hPow x k)
  -/
  rw [pow_apply, if_neg (not_and_of_not_right _ _)]
  /-
    k x✝ x : Nat
    hx : Membership.mem x✝.divisors x
    ⊢ Not (Eq x 0)
  -/
  contrapose! hx
  /-
    k x✝ x : Nat
    hx : Eq x 0
    ⊢ Not (Membership.mem x✝.divisors x)
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


@[arith_mult]
theorem isMultiplicative_one [MonoidWithZero R] : IsMultiplicative (1 : ArithmeticFunction R) :=
  IsMultiplicative.iff_ne_zero.2
        /-
          R : Type u_1
          inst✝ : MonoidWithZero R
          ⊢ Eq (1 1) 1
        -/
    ⟨by simp, by
        /-
          🎉 no goals
        -/
      /-
        R : Type u_1
        inst✝ : MonoidWithZero R
        ⊢ ∀ {m n : Nat}, Ne m 0 → Ne n 0 → m.Coprime n → Eq (1 (HMul.hMul m n)) (HMul. …
      -/
      intro m n hm _hn hmn
      /-
        R : Type u_1
        inst✝ : MonoidWithZero R
        m n : Nat
        hm : Ne m 0
        _hn : Ne n 0
        hmn : m.Coprime n
        ⊢ Eq (1 (HMul.hMul m n)) (HMul.hMul (1 m) (1 n))
      -/
      rcases eq_or_ne m 1 with (rfl | hm')
        /-
          case inl
          R : Type u_1
          inst✝ : MonoidWithZero R
          n : Nat
          _hn : Ne n 0
          hm : Ne 1 0
          hmn : Nat.Coprime 1 n
          ⊢ Eq (1 (HMul.hMul 1 n)) (HMul.hMul (1 1) (1 n))
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inr
        R : Type u_1
        inst✝ : MonoidWithZero R
        m n : Nat
        hm : Ne m 0
        _hn : Ne n 0
        hmn : m.Coprime n
        hm' : Ne m 1
        ⊢ Eq (1 (HMul.hMul m n)) (HMul.hMul (1 m) (1 n))
      -/
      rw [one_apply_ne, one_apply_ne hm', zero_mul]
      /-
        case inr
        R : Type u_1
        inst✝ : MonoidWithZero R
        m n : Nat
        hm : Ne m 0
        _hn : Ne n 0
        hmn : m.Coprime n
        hm' : Ne m 1
        ⊢ Ne (HMul.hMul m n) 1
      -/
      rw [Ne, mul_eq_one, not_and_or]
      /-
        case inr
        R : Type u_1
        inst✝ : MonoidWithZero R
        m n : Nat
        hm : Ne m 0
        _hn : Ne n 0
        hmn : m.Coprime n
        hm' : Ne m 1
        ⊢ Or (Not (Eq m 1)) (Not (Eq n 1))
      -/
      exact Or.inl hm'⟩
      /-
        🎉 no goals
      -/


@[arith_mult]
theorem isMultiplicative_zeta : IsMultiplicative ζ :=
                                     /-
                                       ⊢ Eq (ArithmeticFunction.zeta 1) 1
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  IsMultiplicative.iff_ne_zero.2 ⟨by simp, by simp +contextual⟩
                                              /-
                                                🎉 no goals
                                              -/


@[arith_mult]
theorem isMultiplicative_id : IsMultiplicative ArithmeticFunction.id :=
  ⟨rfl, fun {_ _} _ => rfl⟩


@[arith_mult]
theorem IsMultiplicative.ppow [CommSemiring R] {f : ArithmeticFunction R} (hf : f.IsMultiplicative)
    {k : ℕ} : IsMultiplicative (f.ppow k) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    k : Nat
    ⊢ (f.ppow k).IsMultiplicative
  -/
  induction' k with k hi
    /-
      case zero
      R : Type u_1
      inst✝ : CommSemiring R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      ⊢ (f.ppow 0).IsMultiplicative
    -/
  · exact isMultiplicative_zeta.natCast
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : CommSemiring R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      k : Nat
      hi : (f.ppow k).IsMultiplicative
      ⊢ (f.ppow (HAdd.hAdd k 1)).IsMultiplicative
    -/
  · rw [ppow_succ']
    /-
      case succ
      R : Type u_1
      inst✝ : CommSemiring R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      k : Nat
      hi : (f.ppow k).IsMultiplicative
      ⊢ (f.pmul (f.ppow k)).IsMultiplicative
    -/
    apply hf.pmul hi
    /-
      🎉 no goals
    -/


@[arith_mult]
theorem isMultiplicative_pow {k : ℕ} : IsMultiplicative (pow k) :=
  isMultiplicative_id.ppow


@[arith_mult]
theorem isMultiplicative_sigma {k : ℕ} : IsMultiplicative (σ k) := by
  /-
    k : Nat
    ⊢ (ArithmeticFunction.sigma k).IsMultiplicative
  -/
  rw [← zeta_mul_pow_eq_sigma]
  /-
    k : Nat
    ⊢ (HMul.hMul ArithmeticFunction.zeta (ArithmeticFunction.pow k)).IsMultiplicat …
  -/
  apply isMultiplicative_zeta.mul isMultiplicative_pow
  /-
    🎉 no goals
  -/


/-- `Ω n` is the number of prime factors of `n`. -/
def cardFactors : ArithmeticFunction ℕ :=
                                          /-
                                            R : Type u_1
                                            ⊢ Eq ((fun n => n.primeFactorsList.length) 0) 0
                                          -/
  ⟨fun n => n.primeFactorsList.length, by simp⟩
                                          /-
                                            🎉 no goals
                                          -/


@[inherit_doc]
scoped[ArithmeticFunction] notation "Ω" => ArithmeticFunction.cardFactors


@[inherit_doc]
scoped[ArithmeticFunction.Omega] notation "Ω" => ArithmeticFunction.cardFactors


theorem cardFactors_apply {n : ℕ} : Ω n = n.primeFactorsList.length :=
  rfl


                                       /-
                                         ⊢ Eq (ArithmeticFunction.cardFactors 0) 0
                                       -/
lemma cardFactors_zero : Ω 0 = 0 := by simp
                                       /-
                                         🎉 no goals
                                       -/


                                                /-
                                                  ⊢ Eq (ArithmeticFunction.cardFactors 1) 0
                                                -/
@[simp] theorem cardFactors_one : Ω 1 = 0 := by simp [cardFactors_apply]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem cardFactors_eq_one_iff_prime {n : ℕ} : Ω n = 1 ↔ n.Prime := by
  /-
    n : Nat
    ⊢ Iff (Eq (ArithmeticFunction.cardFactors n) 1) (Nat.Prime n)
  -/
  refine ⟨fun h => ?_, fun h => List.length_eq_one.2 ⟨n, primeFactorsList_prime h⟩⟩
  /-
    n : Nat
    h : Eq (ArithmeticFunction.cardFactors n) 1
    ⊢ Nat.Prime n
  -/
  cases' n with n
    /-
      case zero
      h : Eq (ArithmeticFunction.cardFactors 0) 1
      ⊢ Nat.Prime 0
    -/
  · simp at h
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    h : Eq (ArithmeticFunction.cardFactors (HAdd.hAdd n 1)) 1
    ⊢ Nat.Prime (HAdd.hAdd n 1)
  -/
  rcases List.length_eq_one.1 h with ⟨x, hx⟩
  /-
    case succ.intro
    n : Nat
    h : Eq (ArithmeticFunction.cardFactors (HAdd.hAdd n 1)) 1
    x : Nat
    hx : Eq (HAdd.hAdd n 1).primeFactorsList (List.cons x List.nil)
    ⊢ Nat.Prime (HAdd.hAdd n 1)
  -/
  rw [← prod_primeFactorsList n.add_one_ne_zero, hx, List.prod_singleton]
  /-
    case succ.intro
    n : Nat
    h : Eq (ArithmeticFunction.cardFactors (HAdd.hAdd n 1)) 1
    x : Nat
    hx : Eq (HAdd.hAdd n 1).primeFactorsList (List.cons x List.nil)
    ⊢ Nat.Prime x
  -/
  apply prime_of_mem_primeFactorsList
  /-
    case succ.intro.a
    n : Nat
    h : Eq (ArithmeticFunction.cardFactors (HAdd.hAdd n 1)) 1
    x : Nat
    hx : Eq (HAdd.hAdd n 1).primeFactorsList (List.cons x List.nil)
    ⊢ Membership.mem (Nat.primeFactorsList ?succ.intro.n) x
  -/
  rw [hx, List.mem_singleton]
  /-
    🎉 no goals
  -/


theorem cardFactors_mul {m n : ℕ} (m0 : m ≠ 0) (n0 : n ≠ 0) : Ω (m * n) = Ω m + Ω n := by
  rw [cardFactors_apply, cardFactors_apply, cardFactors_apply, ← Multiset.coe_card, ← factors_eq,
    UniqueFactorizationMonoid.normalizedFactors_mul m0 n0, factors_eq, factors_eq,
    Multiset.card_add, Multiset.coe_card, Multiset.coe_card]


theorem cardFactors_multiset_prod {s : Multiset ℕ} (h0 : s.prod ≠ 0) :
    Ω s.prod = (Multiset.map Ω s).sum := by
  induction s using Multiset.induction_on with
  | empty => simp
  | cons ih => simp_all [cardFactors_mul, not_or]


@[simp]
theorem cardFactors_apply_prime {p : ℕ} (hp : p.Prime) : Ω p = 1 :=
  cardFactors_eq_one_iff_prime.2 hp


@[simp]
theorem cardFactors_apply_prime_pow {p k : ℕ} (hp : p.Prime) : Ω (p ^ k) = k := by
  /-
    p k : Nat
    hp : Nat.Prime p
    ⊢ Eq (ArithmeticFunction.cardFactors (HPow.hPow p k)) k
  -/
  rw [cardFactors_apply, hp.primeFactorsList_pow, List.length_replicate]
  /-
    🎉 no goals
  -/


/-- `ω n` is the number of distinct prime factors of `n`. -/
def cardDistinctFactors : ArithmeticFunction ℕ :=
                                                /-
                                                  R : Type u_1
                                                  ⊢ Eq ((fun n => n.primeFactorsList.dedup.length) 0) 0
                                                -/
  ⟨fun n => n.primeFactorsList.dedup.length, by simp⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[inherit_doc]
scoped[ArithmeticFunction] notation "ω" => ArithmeticFunction.cardDistinctFactors


@[inherit_doc]
scoped[ArithmeticFunction.omega] notation "ω" => ArithmeticFunction.cardDistinctFactors


                                                 /-
                                                   ⊢ Eq (ArithmeticFunction.cardDistinctFactors 0) 0
                                                 -/
theorem cardDistinctFactors_zero : ω 0 = 0 := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                /-
                                                  ⊢ Eq (ArithmeticFunction.cardDistinctFactors 1) 0
                                                -/
theorem cardDistinctFactors_one : ω 1 = 0 := by simp [cardDistinctFactors]
                                                /-
                                                  🎉 no goals
                                                -/


theorem cardDistinctFactors_apply {n : ℕ} : ω n = n.primeFactorsList.dedup.length :=
  rfl


theorem cardDistinctFactors_eq_cardFactors_iff_squarefree {n : ℕ} (h0 : n ≠ 0) :
    ω n = Ω n ↔ Squarefree n := by
  /-
    n : Nat
    h0 : Ne n 0
    ⊢ Iff (Eq (ArithmeticFunction.cardDistinctFactors n) (ArithmeticFunction.cardF …
  -/
  rw [squarefree_iff_nodup_primeFactorsList h0, cardDistinctFactors_apply]
  /-
    n : Nat
    h0 : Ne n 0
    ⊢ Iff (Eq n.primeFactorsList.dedup.length (ArithmeticFunction.cardFactors n))  …
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      h0 : Ne n 0
      h : Eq n.primeFactorsList.dedup.length (ArithmeticFunction.cardFactors n)
      ⊢ n.primeFactorsList.Nodup
    -/
  · rw [← n.primeFactorsList.dedup_sublist.eq_of_length h]
    /-
      case mp
      n : Nat
      h0 : Ne n 0
      h : Eq n.primeFactorsList.dedup.length (ArithmeticFunction.cardFactors n)
      ⊢ n.primeFactorsList.dedup.Nodup
    -/
    apply List.nodup_dedup
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      h0 : Ne n 0
      h : n.primeFactorsList.Nodup
      ⊢ Eq n.primeFactorsList.dedup.length (ArithmeticFunction.cardFactors n)
    -/
  · rw [h.dedup]
    /-
      case mpr
      n : Nat
      h0 : Ne n 0
      h : n.primeFactorsList.Nodup
      ⊢ Eq n.primeFactorsList.length (ArithmeticFunction.cardFactors n)
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem cardDistinctFactors_apply_prime_pow {p k : ℕ} (hp : p.Prime) (hk : k ≠ 0) :
    ω (p ^ k) = 1 := by
  rw [cardDistinctFactors_apply, hp.primeFactorsList_pow, List.replicate_dedup hk,
    List.length_singleton]


@[simp]
theorem cardDistinctFactors_apply_prime {p : ℕ} (hp : p.Prime) : ω p = 1 := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq (ArithmeticFunction.cardDistinctFactors p) 1
  -/
  rw [← pow_one p, cardDistinctFactors_apply_prime_pow hp one_ne_zero]
  /-
    🎉 no goals
  -/


/-- `μ` is the Möbius function. If `n` is squarefree with an even number of distinct prime factors,
  `μ n = 1`. If `n` is squarefree with an odd number of distinct prime factors, `μ n = -1`.
  If `n` is not squarefree, `μ n = 0`. -/
def moebius : ArithmeticFunction ℤ :=
                                                                 /-
                                                                   R : Type u_1
                                                                   ⊢ Eq ((fun n => ite (Squarefree n) (HPow.hPow (-1) (ArithmeticFunction.cardFac …
                                                                 -/
  ⟨fun n => if Squarefree n then (-1) ^ cardFactors n else 0, by simp⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[inherit_doc]
scoped[ArithmeticFunction] notation "μ" => ArithmeticFunction.moebius


@[inherit_doc]
scoped[ArithmeticFunction.Moebius] notation "μ" => ArithmeticFunction.moebius


@[simp]
theorem moebius_apply_of_squarefree {n : ℕ} (h : Squarefree n) : μ n = (-1) ^ cardFactors n :=
  if_pos h


@[simp]
theorem moebius_eq_zero_of_not_squarefree {n : ℕ} (h : ¬Squarefree n) : μ n = 0 :=
  if_neg h


                                          /-
                                            ⊢ Eq (ArithmeticFunction.moebius 1) 1
                                          -/
theorem moebius_apply_one : μ 1 = 1 := by simp
                                          /-
                                            🎉 no goals
                                          -/


theorem moebius_ne_zero_iff_squarefree {n : ℕ} : μ n ≠ 0 ↔ Squarefree n := by
  /-
    n : Nat
    ⊢ Iff (Ne (ArithmeticFunction.moebius n) 0) (Squarefree n)
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      h : Ne (ArithmeticFunction.moebius n) 0
      ⊢ Squarefree n
    -/
  · contrapose! h
    /-
      case mp
      n : Nat
      h : Not (Squarefree n)
      ⊢ Eq (ArithmeticFunction.moebius n) 0
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      h : Squarefree n
      ⊢ Ne (ArithmeticFunction.moebius n) 0
    -/
  · simp [h, pow_ne_zero]
    /-
      🎉 no goals
    -/


theorem moebius_eq_or (n : ℕ) : μ n = 0 ∨ μ n = 1 ∨ μ n = -1 := by
  /-
    n : Nat
    ⊢ Or (Eq (ArithmeticFunction.moebius n) 0) (Or (Eq (ArithmeticFunction.moebius …
  -/
  simp only [moebius, coe_mk]
  /-
    n : Nat
    ⊢ Or (Eq (ite (Squarefree n) (HPow.hPow (-1) (ArithmeticFunction.cardFactors n …
  -/
  split_ifs
    /-
      case pos
      n : Nat
      h✝ : Squarefree n
      ⊢ Or (Eq (HPow.hPow (-1) (ArithmeticFunction.cardFactors n)) 0) (Or (Eq (HPow. …
    -/
  · right
    /-
      case pos.h
      n : Nat
      h✝ : Squarefree n
      ⊢ Or (Eq (HPow.hPow (-1) (ArithmeticFunction.cardFactors n)) 1) (Eq (HPow.hPow …
    -/
    exact neg_one_pow_eq_or ..
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      h✝ : Not (Squarefree n)
      ⊢ Or (Eq 0 0) (Or (Eq 0 1) False)
    -/
  · left
    /-
      case neg.h
      n : Nat
      h✝ : Not (Squarefree n)
      ⊢ Eq 0 0
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem moebius_ne_zero_iff_eq_or {n : ℕ} : μ n ≠ 0 ↔ μ n = 1 ∨ μ n = -1 := by
  /-
    n : Nat
    ⊢ Iff (Ne (ArithmeticFunction.moebius n) 0) (Or (Eq (ArithmeticFunction.moebiu …
  -/
  have := moebius_eq_or n
  /-
    n : Nat
    this : Or (Eq (ArithmeticFunction.moebius n) 0) (Or (Eq (ArithmeticFunction.mo …
    ⊢ Iff (Ne (ArithmeticFunction.moebius n) 0) (Or (Eq (ArithmeticFunction.moebiu …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem moebius_sq_eq_one_of_squarefree {l : ℕ} (hl : Squarefree l) : μ l ^ 2 = 1 := by
  /-
    l : Nat
    hl : Squarefree l
    ⊢ Eq (HPow.hPow (ArithmeticFunction.moebius l) 2) 1
  -/
  rw [moebius_apply_of_squarefree hl, ← pow_mul, mul_comm, pow_mul, neg_one_sq, one_pow]
  /-
    🎉 no goals
  -/


theorem abs_moebius_eq_one_of_squarefree {l : ℕ} (hl : Squarefree l) : |μ l| = 1 := by
  /-
    l : Nat
    hl : Squarefree l
    ⊢ Eq (abs (ArithmeticFunction.moebius l)) 1
  -/
  simp only [moebius_apply_of_squarefree hl, abs_pow, abs_neg, abs_one, one_pow]
  /-
    🎉 no goals
  -/


theorem moebius_sq {n : ℕ} :
    μ n ^ 2 = if Squarefree n then 1 else 0 := by
  /-
    n : Nat
    ⊢ Eq (HPow.hPow (ArithmeticFunction.moebius n) 2) (ite (Squarefree n) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      n : Nat
      h : Squarefree n
      ⊢ Eq (HPow.hPow (ArithmeticFunction.moebius n) 2) 1
    -/
  · exact moebius_sq_eq_one_of_squarefree h
    /-
      🎉 no goals
    -/
  · simp only [pow_eq_zero_iff, moebius_eq_zero_of_not_squarefree h,
    zero_pow (show 2 ≠ 0 by norm_num)]


theorem abs_moebius {n : ℕ} :
    |μ n| = if Squarefree n then 1 else 0 := by
  /-
    n : Nat
    ⊢ Eq (abs (ArithmeticFunction.moebius n)) (ite (Squarefree n) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      n : Nat
      h : Squarefree n
      ⊢ Eq (abs (ArithmeticFunction.moebius n)) 1
    -/
  · exact abs_moebius_eq_one_of_squarefree h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      h : Not (Squarefree n)
      ⊢ Eq (abs (ArithmeticFunction.moebius n)) 0
    -/
  · simp only [moebius_eq_zero_of_not_squarefree h, abs_zero]
    /-
      🎉 no goals
    -/


theorem abs_moebius_le_one {n : ℕ} : |μ n| ≤ 1 := by
  /-
    n : Nat
    ⊢ LE.le (abs (ArithmeticFunction.moebius n)) 1
  -/
  rw [abs_moebius, apply_ite (· ≤ 1)]
  /-
    n : Nat
    ⊢ ite (Squarefree n) (LE.le 1 1) (LE.le 0 1)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem moebius_apply_prime {p : ℕ} (hp : p.Prime) : μ p = -1 := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq (ArithmeticFunction.moebius p) (-1)
  -/
  rw [moebius_apply_of_squarefree hp.squarefree, cardFactors_apply_prime hp, pow_one]
  /-
    🎉 no goals
  -/


theorem moebius_apply_prime_pow {p k : ℕ} (hp : p.Prime) (hk : k ≠ 0) :
    μ (p ^ k) = if k = 1 then -1 else 0 := by
  /-
    p k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    ⊢ Eq (ArithmeticFunction.moebius (HPow.hPow p k)) (ite (Eq k 1) (-1) 0)
  -/
  split_ifs with h
    /-
      case pos
      p k : Nat
      hp : Nat.Prime p
      hk : Ne k 0
      h : Eq k 1
      ⊢ Eq (ArithmeticFunction.moebius (HPow.hPow p k)) (-1)
    -/
  · rw [h, pow_one, moebius_apply_prime hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    p k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    h : Not (Eq k 1)
    ⊢ Eq (ArithmeticFunction.moebius (HPow.hPow p k)) 0
  -/
  rw [moebius_eq_zero_of_not_squarefree]
  /-
    case neg
    p k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    h : Not (Eq k 1)
    ⊢ Not (Squarefree (HPow.hPow p k))
  -/
  rw [squarefree_pow_iff hp.ne_one hk, not_and_or]
  /-
    case neg
    p k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    h : Not (Eq k 1)
    ⊢ Or (Not (Squarefree p)) (Not (Eq k 1))
  -/
  exact Or.inr h
  /-
    🎉 no goals
  -/


theorem moebius_apply_isPrimePow_not_prime {n : ℕ} (hn : IsPrimePow n) (hn' : ¬n.Prime) :
    μ n = 0 := by
  /-
    n : Nat
    hn : IsPrimePow n
    hn' : Not (Nat.Prime n)
    ⊢ Eq (ArithmeticFunction.moebius n) 0
  -/
  obtain ⟨p, k, hp, hk, rfl⟩ := (isPrimePow_nat_iff _).1 hn
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    hn : IsPrimePow (HPow.hPow p k)
    hn' : Not (Nat.Prime (HPow.hPow p k))
    ⊢ Eq (ArithmeticFunction.moebius (HPow.hPow p k)) 0
  -/
  rw [moebius_apply_prime_pow hp hk.ne', if_neg]
  /-
    case intro.intro.intro.intro.hnc
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    hn : IsPrimePow (HPow.hPow p k)
    hn' : Not (Nat.Prime (HPow.hPow p k))
    ⊢ Not (Eq k 1)
  -/
  rintro rfl
  /-
    case intro.intro.intro.intro.hnc
    p : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 1
    hn : IsPrimePow (HPow.hPow p 1)
    hn' : Not (Nat.Prime (HPow.hPow p 1))
    ⊢ False
  -/
  exact hn' (by simpa)
  /-
    🎉 no goals
  -/


@[arith_mult]
theorem isMultiplicative_moebius : IsMultiplicative μ := by
  /-
    ⊢ ArithmeticFunction.moebius.IsMultiplicative
  -/
  rw [IsMultiplicative.iff_ne_zero]
  /-
    ⊢ And (Eq (ArithmeticFunction.moebius 1) 1) (∀ {m n : Nat}, Ne m 0 → Ne n 0 →  …
  -/
  refine ⟨by simp, fun {n m} hn hm hnm => ?_⟩
  simp only [moebius, ZeroHom.coe_mk, coe_mk, ZeroHom.toFun_eq_coe, Eq.ndrec, ZeroHom.coe_mk,
    IsUnit.mul_iff, Nat.isUnit_iff, squarefree_mul hnm, ite_zero_mul_ite_zero,
    cardFactors_mul hn hm, pow_add]


theorem IsMultiplicative.prodPrimeFactors_one_add_of_squarefree [CommSemiring R]
    {f : ArithmeticFunction R} (h_mult : f.IsMultiplicative) {n : ℕ} (hn : Squarefree n) :
    ∏ p ∈ n.primeFactors, (1 + f p) = ∑ d ∈ n.divisors, f d := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : ArithmeticFunction R
    h_mult : f.IsMultiplicative
    n : Nat
    hn : Squarefree n
    ⊢ Eq (n.primeFactors.prod fun p => HAdd.hAdd 1 (f p)) (n.divisors.sum fun d => …
  -/
  trans (∏ᵖ p ∣ n, ((ζ : ArithmeticFunction R) + f) p)
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      f : ArithmeticFunction R
      h_mult : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      ⊢ Eq (n.primeFactors.prod fun p => HAdd.hAdd 1 (f p)) ((ArithmeticFunction.pro …
    -/
  · simp_rw [prodPrimeFactors_apply hn.ne_zero, add_apply, natCoe_apply]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      f : ArithmeticFunction R
      h_mult : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      ⊢ Eq (n.primeFactors.prod fun p => HAdd.hAdd 1 (f p)) (n.primeFactors.prod fun …
    -/
    apply Finset.prod_congr rfl; intro p hp
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      f : ArithmeticFunction R
      h_mult : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      p : Nat
      hp : Membership.mem n.primeFactors p
      ⊢ Eq (HAdd.hAdd 1 (f p)) (HAdd.hAdd (↑(ArithmeticFunction.zeta p)) (f p))
    -/
    rw [zeta_apply_ne (prime_of_mem_primeFactorsList <| List.mem_toFinset.mp hp).ne_zero, cast_one]
    /-
      🎉 no goals
    -/
  rw [isMultiplicative_zeta.natCast.prodPrimeFactors_add_of_squarefree h_mult hn,
    coe_zeta_mul_apply]


theorem IsMultiplicative.prodPrimeFactors_one_sub_of_squarefree [CommRing R]
    (f : ArithmeticFunction R) (hf : f.IsMultiplicative) {n : ℕ} (hn : Squarefree n) :
    ∏ p ∈ n.primeFactors, (1 - f p) = ∑ d ∈ n.divisors, μ d * f d := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : ArithmeticFunction R
    hf : f.IsMultiplicative
    n : Nat
    hn : Squarefree n
    ⊢ Eq (n.primeFactors.prod fun p => HSub.hSub 1 (f p)) (n.divisors.sum fun d => …
  -/
  trans (∏ p ∈ n.primeFactors, (1 + (ArithmeticFunction.pmul (μ : ArithmeticFunction R) f) p))
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      ⊢ Eq (n.primeFactors.prod fun p => HSub.hSub 1 (f p)) (n.primeFactors.prod fun …
    -/
  · apply Finset.prod_congr rfl; intro p hp
    rw [pmul_apply, intCoe_apply, ArithmeticFunction.moebius_apply_prime
        (prime_of_mem_primeFactorsList (List.mem_toFinset.mp hp))]
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      p : Nat
      hp : Membership.mem n.primeFactors p
      ⊢ Eq (HSub.hSub 1 (f p)) (HAdd.hAdd 1 (HMul.hMul (↑(-1)) (f p)))
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      ⊢ Eq (n.primeFactors.prod fun p => HAdd.hAdd 1 (((↑ArithmeticFunction.moebius) …
    -/
  · rw [(isMultiplicative_moebius.intCast.pmul hf).prodPrimeFactors_one_add_of_squarefree hn]
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : ArithmeticFunction R
      hf : f.IsMultiplicative
      n : Nat
      hn : Squarefree n
      ⊢ Eq (n.divisors.sum fun d => ((↑ArithmeticFunction.moebius).pmul f) d) (n.div …
    -/
    simp_rw [pmul_apply, intCoe_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem moebius_mul_coe_zeta : (μ * ζ : ArithmeticFunction ℤ) = 1 := by
  /-
    ⊢ Eq (HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) 1
  -/
  ext n
  /-
    case h
    n : Nat
    ⊢ Eq ((HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) n) (1 n)
  -/
  refine recOnPosPrimePosCoprime ?_ ?_ ?_ ?_ n
    /-
      case h.refine_1
      n : Nat
      ⊢ ∀ (p n : Nat), Nat.Prime p → LT.lt 0 n → Eq ((HMul.hMul ArithmeticFunction.m …
    -/
  · intro p n hp hn
    /-
      case h.refine_1
      n✝ p n : Nat
      hp : Nat.Prime p
      hn : LT.lt 0 n
      ⊢ Eq ((HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) (HPow.hP …
    -/
    rw [coe_mul_zeta_apply, sum_divisors_prime_pow hp, sum_range_succ']
    simp_rw [Nat.pow_zero, moebius_apply_one,
      moebius_apply_prime_pow hp (Nat.succ_ne_zero _), Nat.succ_inj', sum_ite_eq', mem_range,
      if_pos hn, neg_add_cancel]
    /-
      case h.refine_1
      n✝ p n : Nat
      hp : Nat.Prime p
      hn : LT.lt 0 n
      ⊢ Eq 0 (1 (HPow.hPow p n))
    -/
    rw [one_apply_ne]
    /-
      case h.refine_1
      n✝ p n : Nat
      hp : Nat.Prime p
      hn : LT.lt 0 n
      ⊢ Ne (HPow.hPow p n) 1
    -/
    rw [Ne, pow_eq_one_iff]
      /-
        case h.refine_1
        n✝ p n : Nat
        hp : Nat.Prime p
        hn : LT.lt 0 n
        ⊢ Not (Eq p 1)
      -/
    · exact hp.ne_one
      /-
        🎉 no goals
      -/
      /-
        case h.refine_1
        n✝ p n : Nat
        hp : Nat.Prime p
        hn : LT.lt 0 n
        ⊢ Ne n 0
      -/
    · exact hn.ne'
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      n : Nat
      ⊢ Eq ((HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) 0) (1 0)
    -/
  · rw [ZeroHom.map_zero, ZeroHom.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      n : Nat
      ⊢ Eq ((HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) 1) (1 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.refine_4
      n : Nat
      ⊢ ∀ (a b : Nat), LT.lt 1 a → LT.lt 1 b → a.Coprime b → Eq ((HMul.hMul Arithmet …
    -/
  · intro a b _ha _hb hab ha' hb'
    rw [IsMultiplicative.map_mul_of_coprime _ hab, ha', hb',
      IsMultiplicative.map_mul_of_coprime isMultiplicative_one hab]
    /-
      n a b : Nat
      _ha : LT.lt 1 a
      _hb : LT.lt 1 b
      hab : a.Coprime b
      ha' : Eq ((HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) a) ( …
      hb' : Eq ((HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) b) ( …
      ⊢ (HMul.hMul ArithmeticFunction.moebius ↑ArithmeticFunction.zeta).IsMultiplica …
    -/
    exact isMultiplicative_moebius.mul isMultiplicative_zeta.natCast
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_zeta_mul_moebius : (ζ * μ : ArithmeticFunction ℤ) = 1 := by
  /-
    ⊢ Eq (HMul.hMul (↑ArithmeticFunction.zeta) ArithmeticFunction.moebius) 1
  -/
  rw [mul_comm, moebius_mul_coe_zeta]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_moebius_mul_coe_zeta [Ring R] : (μ * ζ : ArithmeticFunction R) = 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq (HMul.hMul ↑ArithmeticFunction.moebius ↑ArithmeticFunction.zeta) 1
  -/
  rw [← coe_coe, ← intCoe_mul, moebius_mul_coe_zeta, intCoe_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_zeta_mul_coe_moebius [Ring R] : (ζ * μ : ArithmeticFunction R) = 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq (HMul.hMul ↑ArithmeticFunction.zeta ↑ArithmeticFunction.moebius) 1
  -/
  rw [← coe_coe, ← intCoe_mul, coe_zeta_mul_moebius, intCoe_one]
  /-
    🎉 no goals
  -/


instance : Invertible (ζ : ArithmeticFunction R) where
  invOf := μ
  invOf_mul_self := coe_moebius_mul_coe_zeta
  mul_invOf_self := coe_zeta_mul_coe_moebius


/-- A unit in `ArithmeticFunction R` that evaluates to `ζ`, with inverse `μ`. -/
def zetaUnit : (ArithmeticFunction R)ˣ :=
  ⟨ζ, μ, coe_zeta_mul_coe_moebius, coe_moebius_mul_coe_zeta⟩


@[simp]
theorem coe_zetaUnit : ((zetaUnit : (ArithmeticFunction R)ˣ) : ArithmeticFunction R) = ζ :=
  rfl


@[simp]
theorem inv_zetaUnit : ((zetaUnit⁻¹ : (ArithmeticFunction R)ˣ) : ArithmeticFunction R) = μ :=
  rfl


/-- Möbius inversion for functions to an `AddCommGroup`. -/
theorem sum_eq_iff_sum_smul_moebius_eq [AddCommGroup R] {f g : ℕ → R} :
    (∀ n > 0, ∑ i ∈ n.divisors, f i = g n) ↔
      ∀ n > 0, ∑ x ∈ n.divisorsAntidiagonal, μ x.fst • g x.snd = f n := by
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    f g : Nat → R
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (g n)) (∀ (n  …
  -/
  let f' : ArithmeticFunction R := ⟨fun x => if x = 0 then 0 else f x, if_pos rfl⟩
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    f g : Nat → R
    f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (g n)) (∀ (n  …
  -/
  let g' : ArithmeticFunction R := ⟨fun x => if x = 0 then 0 else g x, if_pos rfl⟩
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    f g : Nat → R
    f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
    g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (g n)) (∀ (n  …
  -/
  trans (ζ : ArithmeticFunction ℤ) • f' = g'
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (g n)) (Eq (H …
    -/
  · rw [ArithmeticFunction.ext_iff]
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (g n)) (∀ (x  …
    -/
    apply forall_congr'
    /-
      case h
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ ∀ (a : Nat), Iff (GT.gt a 0 → Eq (a.divisors.sum fun i => f i) (g a)) (Eq (( …
    -/
    intro n
    cases n with
    | zero => simp
    | succ n =>
      rw [coe_zeta_smul_apply]
      simp only [n.succ_ne_zero, forall_prop_of_true, succ_pos', if_false, ZeroHom.coe_mk]
      simp only [f', g', coe_mk, succ_ne_zero, ite_false]
      rw [sum_congr rfl fun x hx => ?_]
      rw [if_neg (Nat.pos_of_mem_divisors hx).ne']
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    f g : Nat → R
    f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
    g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
    ⊢ Iff (Eq (HSMul.hSMul (↑ArithmeticFunction.zeta) f') g') (∀ (n : Nat), GT.gt  …
  -/
  trans μ • g' = f'
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ Iff (Eq (HSMul.hSMul (↑ArithmeticFunction.zeta) f') g') (Eq (HSMul.hSMul Ari …
    -/
  · constructor <;> intro h
      /-
        case mp
        R : Type u_1
        inst✝ : AddCommGroup R
        f g : Nat → R
        f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
        g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
        h : Eq (HSMul.hSMul (↑ArithmeticFunction.zeta) f') g'
        ⊢ Eq (HSMul.hSMul ArithmeticFunction.moebius g') f'
      -/
    · rw [← h, ← mul_smul, moebius_mul_coe_zeta, one_smul]
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_1
        inst✝ : AddCommGroup R
        f g : Nat → R
        f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
        g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
        h : Eq (HSMul.hSMul ArithmeticFunction.moebius g') f'
        ⊢ Eq (HSMul.hSMul (↑ArithmeticFunction.zeta) f') g'
      -/
    · rw [← h, ← mul_smul, coe_zeta_mul_moebius, one_smul]
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ Iff (Eq (HSMul.hSMul ArithmeticFunction.moebius g') f') (∀ (n : Nat), GT.gt  …
    -/
  · rw [ArithmeticFunction.ext_iff]
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ Iff (∀ (x : Nat), Eq ((HSMul.hSMul ArithmeticFunction.moebius g') x) (f' x)) …
    -/
    apply forall_congr'
    /-
      case h
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      f' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (f x), map_zer …
      g' : ArithmeticFunction R := { toFun := fun x => ite (Eq x 0) 0 (g x), map_zer …
      ⊢ ∀ (a : Nat), Iff (Eq ((HSMul.hSMul ArithmeticFunction.moebius g') a) (f' a)) …
    -/
    intro n
    cases n with
    | zero => simp
    | succ n =>
      simp only [n.succ_ne_zero, forall_prop_of_true, succ_pos', smul_apply, if_false,
        ZeroHom.coe_mk]
      -- Porting note: added following `simp only`
      simp only [f', g', Nat.isUnit_iff, coe_mk, ZeroHom.toFun_eq_coe, succ_ne_zero, ite_false]
      rw [sum_congr rfl fun x hx => ?_]
      rw [if_neg (Nat.pos_of_mem_divisors (snd_mem_divisors_of_mem_antidiagonal hx)).ne']


/-- Möbius inversion for functions to a `Ring`. -/
theorem sum_eq_iff_sum_mul_moebius_eq [Ring R] {f g : ℕ → R} :
    (∀ n > 0, ∑ i ∈ n.divisors, f i = g n) ↔
      ∀ n > 0, ∑ x ∈ n.divisorsAntidiagonal, (μ x.fst : R) * g x.snd = f n := by
  /-
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (g n)) (∀ (n  …
  -/
  rw [sum_eq_iff_sum_smul_moebius_eq]
  /-
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Eq (n.divisorsAntidiagonal.sum fun x => HSMul. …
  -/
  apply forall_congr'
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    ⊢ ∀ (a : Nat), Iff (GT.gt a 0 → Eq (a.divisorsAntidiagonal.sum fun x => HSMul. …
  -/
  refine fun a => imp_congr_right fun _ => (sum_congr rfl fun x _hx => ?_).congr_left
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    a : Nat
    x✝ : GT.gt a 0
    x : Prod Nat Nat
    _hx : Membership.mem a.divisorsAntidiagonal x
    ⊢ Eq (HSMul.hSMul (ArithmeticFunction.moebius x.1) (g x.2)) (HMul.hMul (↑(Arit …
  -/
  rw [zsmul_eq_mul]
  /-
    🎉 no goals
  -/


/-- Möbius inversion for functions to a `CommGroup`. -/
theorem prod_eq_iff_prod_pow_moebius_eq [CommGroup R] {f g : ℕ → R} :
    (∀ n > 0, ∏ i ∈ n.divisors, f i = g n) ↔
      ∀ n > 0, ∏ x ∈ n.divisorsAntidiagonal, g x.snd ^ μ x.fst = f n :=
  @sum_eq_iff_sum_smul_moebius_eq (Additive R) _ _ _


/-- Möbius inversion for functions to a `CommGroupWithZero`. -/
theorem prod_eq_iff_prod_pow_moebius_eq_of_nonzero [CommGroupWithZero R] {f g : ℕ → R}
    (hf : ∀ n : ℕ, 0 < n → f n ≠ 0) (hg : ∀ n : ℕ, 0 < n → g n ≠ 0) :
    (∀ n > 0, ∏ i ∈ n.divisors, f i = g n) ↔
      ∀ n > 0, ∏ x ∈ n.divisorsAntidiagonal, g x.snd ^ μ x.fst = f n := by
  refine
      Iff.trans
        (Iff.trans (forall_congr' fun n => ?_)
          (@prod_eq_iff_prod_pow_moebius_eq Rˣ _
            (fun n => if h : 0 < n then Units.mk0 (f n) (hf n h) else 1) fun n =>
            if h : 0 < n then Units.mk0 (g n) (hg n h) else 1))
        (forall_congr' fun n => ?_) <;>
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : Nat → R
      hf : ∀ (n : Nat), LT.lt 0 n → Ne (f n) 0
      hg : ∀ (n : Nat), LT.lt 0 n → Ne (g n) 0
      n : Nat
      ⊢ Iff (GT.gt n 0 → Eq (n.divisors.prod fun i => f i) (g n)) (GT.gt n 0 → Eq (n …
    -/
    refine imp_congr_right fun hn => ?_
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : Nat → R
      hf : ∀ (n : Nat), LT.lt 0 n → Ne (f n) 0
      hg : ∀ (n : Nat), LT.lt 0 n → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ Iff (Eq (n.divisors.prod fun i => f i) (g n)) (Eq (n.divisors.prod fun i =>  …
    -/
  · dsimp
    rw [dif_pos hn, ← Units.eq_iff, ← Units.coeHom_apply, map_prod, Units.val_mk0,
      prod_congr rfl _]
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : Nat → R
      hf : ∀ (n : Nat), LT.lt 0 n → Ne (f n) 0
      hg : ∀ (n : Nat), LT.lt 0 n → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ ∀ (x : Nat), Membership.mem n.divisors x → Eq (f x) ((Units.coeHom R) (dite  …
    -/
    intro x hx
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : Nat → R
      hf : ∀ (n : Nat), LT.lt 0 n → Ne (f n) 0
      hg : ∀ (n : Nat), LT.lt 0 n → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      x : Nat
      hx : Membership.mem n.divisors x
      ⊢ Eq (f x) ((Units.coeHom R) (dite (LT.lt 0 x) (fun h => Units.mk0 (f x) ⋯) fu …
    -/
    rw [dif_pos (Nat.pos_of_mem_divisors hx), Units.coeHom_apply, Units.val_mk0]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : Nat → R
      hf : ∀ (n : Nat), LT.lt 0 n → Ne (f n) 0
      hg : ∀ (n : Nat), LT.lt 0 n → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ Iff (Eq (n.divisorsAntidiagonal.prod fun x => HPow.hPow (dite (LT.lt 0 x.2)  …
    -/
  · dsimp
    rw [dif_pos hn, ← Units.eq_iff, ← Units.coeHom_apply, map_prod, Units.val_mk0,
      prod_congr rfl _]
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      f g : Nat → R
      hf : ∀ (n : Nat), LT.lt 0 n → Ne (f n) 0
      hg : ∀ (n : Nat), LT.lt 0 n → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem n.divisorsAntidiagonal x → Eq ((Units.c …
    -/
    intro x hx
    rw [dif_pos (Nat.pos_of_mem_divisors (Nat.snd_mem_divisors_of_mem_antidiagonal hx)),
      Units.coeHom_apply, Units.val_zpow_eq_zpow_val, Units.val_mk0]


/-- Möbius inversion for functions to an `AddCommGroup`, where the equalities only hold on a
well-behaved set. -/
theorem sum_eq_iff_sum_smul_moebius_eq_on [AddCommGroup R] {f g : ℕ → R}
    (s : Set ℕ) (hs : ∀ m n, m ∣ n → n ∈ s → m ∈ s) :
    (∀ n > 0, n ∈ s → (∑ i ∈ n.divisors, f i) = g n) ↔
      ∀ n > 0, n ∈ s → (∑ x ∈ n.divisorsAntidiagonal, μ x.fst • g x.snd) = f n := by
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      ⊢ (∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i => f …
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i =>  …
      ⊢ ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.sum …
    -/
    let G := fun (n : ℕ) => (∑ i ∈ n.divisors, f i)
    /-
      case mp
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i =>  …
      G : Nat → R := fun n => n.divisors.sum fun i => f i
      ⊢ ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.sum …
    -/
    intro n hn hnP
    suffices ∑ d ∈ n.divisors, μ (n/d) • G d = f n from by
      rw [Nat.sum_divisorsAntidiagonal' (f := fun x y => μ x • g y), ← this, sum_congr rfl]
      intro d hd
      rw [← h d (Nat.pos_of_mem_divisors hd) <| hs d n (Nat.dvd_of_mem_divisors hd) hnP]
    /-
      case mp
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i =>  …
      G : Nat → R := fun n => n.divisors.sum fun i => f i
      n : Nat
      hn : GT.gt n 0
      hnP : Membership.mem s n
      ⊢ Eq (n.divisors.sum fun d => HSMul.hSMul (ArithmeticFunction.moebius (HDiv.hD …
    -/
    rw [← Nat.sum_divisorsAntidiagonal' (f := fun x y => μ x • G y)]
    /-
      case mp
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i =>  …
      G : Nat → R := fun n => n.divisors.sum fun i => f i
      n : Nat
      hn : GT.gt n 0
      hnP : Membership.mem s n
      ⊢ Eq (n.divisorsAntidiagonal.sum fun i => HSMul.hSMul (ArithmeticFunction.moeb …
    -/
    apply sum_eq_iff_sum_smul_moebius_eq.mp _ n hn
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i =>  …
      G : Nat → R := fun n => n.divisors.sum fun i => f i
      n : Nat
      hn : GT.gt n 0
      hnP : Membership.mem s n
      ⊢ ∀ (n : Nat), GT.gt n 0 → Eq (n.divisors.sum fun i => f i) (G n)
    -/
    intro _ _; rfl
               /-
                 🎉 no goals
               -/
    /-
      case mpr
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      ⊢ (∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.su …
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.s …
      ⊢ ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i => f  …
    -/
    let F := fun (n : ℕ) => ∑ x ∈ n.divisorsAntidiagonal, μ x.fst • g x.snd
    /-
      case mpr
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.s …
      F : Nat → R := fun n => n.divisorsAntidiagonal.sum fun x => HSMul.hSMul (Arith …
      ⊢ ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i => f  …
    -/
    intro n hn hnP
    suffices ∑ d ∈ n.divisors, F d = g n from by
      rw [← this, sum_congr rfl]
      intro d hd
      rw [← h d (Nat.pos_of_mem_divisors hd) <| hs d n (Nat.dvd_of_mem_divisors hd) hnP]
    /-
      case mpr
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.s …
      F : Nat → R := fun n => n.divisorsAntidiagonal.sum fun x => HSMul.hSMul (Arith …
      n : Nat
      hn : GT.gt n 0
      hnP : Membership.mem s n
      ⊢ Eq (n.divisors.sum fun d => F d) (g n)
    -/
    apply sum_eq_iff_sum_smul_moebius_eq.mpr _ n hn
    /-
      R : Type u_1
      inst✝ : AddCommGroup R
      f g : Nat → R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      h : ∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagonal.s …
      F : Nat → R := fun n => n.divisorsAntidiagonal.sum fun x => HSMul.hSMul (Arith …
      n : Nat
      hn : GT.gt n 0
      hnP : Membership.mem s n
      ⊢ ∀ (n : Nat), GT.gt n 0 → Eq (n.divisorsAntidiagonal.sum fun x => HSMul.hSMul …
    -/
    intro _ _; rfl
               /-
                 🎉 no goals
               -/


theorem sum_eq_iff_sum_smul_moebius_eq_on' [AddCommGroup R] {f g : ℕ → R}
    (s : Set ℕ) (hs : ∀ m n, m ∣ n → n ∈ s → m ∈ s) (hs₀ : 0 ∉ s) :
    (∀ n ∈ s, (∑ i ∈ n.divisors, f i) = g n) ↔
     ∀ n ∈ s, (∑ x ∈ n.divisorsAntidiagonal, μ x.fst • g x.snd) = f n := by
  have : ∀ P : ℕ → Prop, ((∀ n ∈ s, P n) ↔ (∀ n > 0, n ∈ s → P n)) := fun P ↦ by
    refine forall_congr' (fun n ↦ ⟨fun h _ ↦ h, fun h hn ↦ h ?_ hn⟩)
    contrapose! hs₀
    simpa [nonpos_iff_eq_zero.mp hs₀] using hn
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    hs₀ : Not (Membership.mem s 0)
    this : ∀ (P : Nat → Prop), Iff (∀ (n : Nat), Membership.mem s n → P n) (∀ (n : …
    ⊢ Iff (∀ (n : Nat), Membership.mem s n → Eq (n.divisors.sum fun i => f i) (g n …
  -/
  simpa only [this] using sum_eq_iff_sum_smul_moebius_eq_on s hs
  /-
    🎉 no goals
  -/


/-- Möbius inversion for functions to a `Ring`, where the equalities only hold on a well-behaved
set. -/
theorem sum_eq_iff_sum_mul_moebius_eq_on [Ring R] {f g : ℕ → R}
    (s : Set ℕ) (hs : ∀ m n, m ∣ n → n ∈ s → m ∈ s) :
    (∀ n > 0, n ∈ s → (∑ i ∈ n.divisors, f i) = g n) ↔
      ∀ n > 0, n ∈ s →
        (∑ x ∈ n.divisorsAntidiagonal, (μ x.fst : R) * g x.snd) = f n := by
  /-
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisors.sum fun i  …
  -/
  rw [sum_eq_iff_sum_smul_moebius_eq_on s hs]
  /-
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    ⊢ Iff (∀ (n : Nat), GT.gt n 0 → Membership.mem s n → Eq (n.divisorsAntidiagona …
  -/
  apply forall_congr'
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    ⊢ ∀ (a : Nat), Iff (GT.gt a 0 → Membership.mem s a → Eq (a.divisorsAntidiagona …
  -/
  intro a; refine imp_congr_right ?_
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    a : Nat
    ⊢ GT.gt a 0 → Iff (Membership.mem s a → Eq (a.divisorsAntidiagonal.sum fun x = …
  -/
  refine fun _ => imp_congr_right fun _ => (sum_congr rfl fun x _hx => ?_).congr_left
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    f g : Nat → R
    s : Set Nat
    hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
    a : Nat
    x✝¹ : GT.gt a 0
    x✝ : Membership.mem s a
    x : Prod Nat Nat
    _hx : Membership.mem a.divisorsAntidiagonal x
    ⊢ Eq (HSMul.hSMul (ArithmeticFunction.moebius x.1) (g x.2)) (HMul.hMul (↑(Arit …
  -/
  rw [zsmul_eq_mul]
  /-
    🎉 no goals
  -/


/-- Möbius inversion for functions to a `CommGroup`, where the equalities only hold on a
well-behaved set. -/
theorem prod_eq_iff_prod_pow_moebius_eq_on [CommGroup R] {f g : ℕ → R}
    (s : Set ℕ) (hs : ∀ m n, m ∣ n → n ∈ s → m ∈ s) :
    (∀ n > 0, n ∈ s → (∏ i ∈ n.divisors, f i) = g n) ↔
      ∀ n > 0, n ∈ s → (∏ x ∈ n.divisorsAntidiagonal, g x.snd ^ μ x.fst) = f n :=
  @sum_eq_iff_sum_smul_moebius_eq_on (Additive R) _ _ _ s hs


/-- Möbius inversion for functions to a `CommGroupWithZero`, where the equalities only hold on
a well-behaved set. -/
theorem prod_eq_iff_prod_pow_moebius_eq_on_of_nonzero [CommGroupWithZero R]
    (s : Set ℕ) (hs : ∀ m n, m ∣ n → n ∈ s → m ∈ s) {f g : ℕ → R}
    (hf : ∀ n > 0, f n ≠ 0) (hg : ∀ n > 0, g n ≠ 0) :
    (∀ n > 0, n ∈ s → (∏ i ∈ n.divisors, f i) = g n) ↔
      ∀ n > 0, n ∈ s → (∏ x ∈ n.divisorsAntidiagonal, g x.snd ^ μ x.fst) = f n := by
  refine
      Iff.trans
        (Iff.trans (forall_congr' fun n => ?_)
          (@prod_eq_iff_prod_pow_moebius_eq_on Rˣ _
            (fun n => if h : 0 < n then Units.mk0 (f n) (hf n h) else 1)
            (fun n => if h : 0 < n then Units.mk0 (g n) (hg n h) else 1)
            s hs) )
        (forall_congr' fun n => ?_) <;>
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommGroupWithZero R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      f g : Nat → R
      hf : ∀ (n : Nat), GT.gt n 0 → Ne (f n) 0
      hg : ∀ (n : Nat), GT.gt n 0 → Ne (g n) 0
      n : Nat
      ⊢ Iff (GT.gt n 0 → Membership.mem s n → Eq (n.divisors.prod fun i => f i) (g n …
    -/
    refine imp_congr_right fun hn => ?_
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommGroupWithZero R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      f g : Nat → R
      hf : ∀ (n : Nat), GT.gt n 0 → Ne (f n) 0
      hg : ∀ (n : Nat), GT.gt n 0 → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ Iff (Membership.mem s n → Eq (n.divisors.prod fun i => f i) (g n)) (Membersh …
    -/
  · dsimp
    rw [dif_pos hn, ← Units.eq_iff, ← Units.coeHom_apply, map_prod, Units.val_mk0,
      prod_congr rfl _]
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      f g : Nat → R
      hf : ∀ (n : Nat), GT.gt n 0 → Ne (f n) 0
      hg : ∀ (n : Nat), GT.gt n 0 → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ ∀ (x : Nat), Membership.mem n.divisors x → Eq (f x) ((Units.coeHom R) (dite  …
    -/
    intro x hx
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      f g : Nat → R
      hf : ∀ (n : Nat), GT.gt n 0 → Ne (f n) 0
      hg : ∀ (n : Nat), GT.gt n 0 → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      x : Nat
      hx : Membership.mem n.divisors x
      ⊢ Eq (f x) ((Units.coeHom R) (dite (LT.lt 0 x) (fun h => Units.mk0 (f x) ⋯) fu …
    -/
    rw [dif_pos (Nat.pos_of_mem_divisors hx), Units.coeHom_apply, Units.val_mk0]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommGroupWithZero R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      f g : Nat → R
      hf : ∀ (n : Nat), GT.gt n 0 → Ne (f n) 0
      hg : ∀ (n : Nat), GT.gt n 0 → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ Iff (Membership.mem s n → Eq (n.divisorsAntidiagonal.prod fun x => HPow.hPow …
    -/
  · dsimp
    rw [dif_pos hn, ← Units.eq_iff, ← Units.coeHom_apply, map_prod, Units.val_mk0,
      prod_congr rfl _]
    /-
      R : Type u_1
      inst✝ : CommGroupWithZero R
      s : Set Nat
      hs : ∀ (m n : Nat), Dvd.dvd m n → Membership.mem s n → Membership.mem s m
      f g : Nat → R
      hf : ∀ (n : Nat), GT.gt n 0 → Ne (f n) 0
      hg : ∀ (n : Nat), GT.gt n 0 → Ne (g n) 0
      n : Nat
      hn : GT.gt n 0
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem n.divisorsAntidiagonal x → Eq ((Units.c …
    -/
    intro x hx
    rw [dif_pos (Nat.pos_of_mem_divisors (Nat.snd_mem_divisors_of_mem_antidiagonal hx)),
      Units.coeHom_apply, Units.val_zpow_eq_zpow_val, Units.val_mk0]


theorem _root_.Nat.card_divisors {n : ℕ} (hn : n ≠ 0) :
    #n.divisors = n.primeFactors.prod (n.factorization · + 1) := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq n.divisors.card (n.primeFactors.prod fun x => HAdd.hAdd (n.factorization  …
  -/
  rw [← sigma_zero_apply, isMultiplicative_sigma.multiplicative_factorization _ hn]
  exact Finset.prod_congr n.support_factorization fun _ h =>
    sigma_zero_apply_prime_pow <| Nat.prime_of_mem_primeFactors h


@[deprecated "No deprecation message was provided." (since := "2024-06-09")]
theorem card_divisors (n : ℕ) (hn : n ≠ 0) :
    #n.divisors = n.primeFactors.prod (n.factorization · + 1) := Nat.card_divisors hn


theorem _root_.Nat.sum_divisors {n : ℕ} (hn : n ≠ 0) :
    ∑ d ∈ n.divisors, d = ∏ p ∈ n.primeFactors, ∑ k ∈ .range (n.factorization p + 1), p ^ k := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (n.divisors.sum fun d => d) (n.primeFactors.prod fun p => (Finset.range ( …
  -/
  rw [← sigma_one_apply, isMultiplicative_sigma.multiplicative_factorization _ hn]
  exact Finset.prod_congr n.support_factorization fun _ h =>
    sigma_one_apply_prime_pow <| Nat.prime_of_mem_primeFactors h


theorem card_divisors_mul {m n : ℕ} (hmn : m.Coprime n) :
    #(m * n).divisors = #m.divisors * #n.divisors := by
  /-
    m n : Nat
    hmn : m.Coprime n
    ⊢ Eq (HMul.hMul m n).divisors.card (HMul.hMul m.divisors.card n.divisors.card)
  -/
  simp only [← sigma_zero_apply, isMultiplicative_sigma.map_mul_of_coprime hmn]
  /-
    🎉 no goals
  -/


theorem sum_divisors_mul {m n : ℕ} (hmn : m.Coprime n) :
    ∑ d ∈ (m * n).divisors, d = (∑ d ∈ m.divisors, d) * ∑ d ∈ n.divisors, d := by
  /-
    m n : Nat
    hmn : m.Coprime n
    ⊢ Eq ((HMul.hMul m n).divisors.sum fun d => d) (HMul.hMul (m.divisors.sum fun  …
  -/
  simp only [← sigma_one_apply, isMultiplicative_sigma.map_mul_of_coprime hmn]
  /-
    🎉 no goals
  -/


