/-- The left ideal of matrices with entries in `I ≤ R`. -/
def matricesOver (I : Ideal R) : Ideal (Matrix n n R) where
  carrier := { M | ∀ i j, M i j ∈ I }
  add_mem' ha hb i j := I.add_mem (ha i j) (hb i j)
  zero_mem' _ _ := I.zero_mem
  smul_mem' M N hN := by
    /-
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      M N : Matrix n n R
      hN : Membership.mem { carrier := setOf fun M => ∀ (i j : n), Membership.mem I  …
      ⊢ Membership.mem { carrier := setOf fun M => ∀ (i j : n), Membership.mem I (M  …
    -/
    intro i j
    /-
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      M N : Matrix n n R
      hN : Membership.mem { carrier := setOf fun M => ∀ (i j : n), Membership.mem I  …
      i j : n
      ⊢ Membership.mem I (HSMul.hSMul M N i j)
    -/
    rw [smul_eq_mul, mul_apply]
    /-
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      M N : Matrix n n R
      hN : Membership.mem { carrier := setOf fun M => ∀ (i j : n), Membership.mem I  …
      i j : n
      ⊢ Membership.mem I (Finset.univ.sum fun j_1 => HMul.hMul (M i j_1) (N j_1 j))
    -/
    apply sum_mem
    /-
      case h
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      M N : Matrix n n R
      hN : Membership.mem { carrier := setOf fun M => ∀ (i j : n), Membership.mem I  …
      i j : n
      ⊢ ∀ (c : n), Membership.mem Finset.univ c → Membership.mem I (HMul.hMul (M i c …
    -/
    intro k _
    /-
      case h
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      M N : Matrix n n R
      hN : Membership.mem { carrier := setOf fun M => ∀ (i j : n), Membership.mem I  …
      i j k : n
      a✝ : Membership.mem Finset.univ k
      ⊢ Membership.mem I (HMul.hMul (M i k) (N k j))
    -/
    apply I.mul_mem_left _ (hN k j)
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_matricesOver (I : Ideal R) (M : Matrix n n R) :
                                                  /-
                                                    R : Type u_1
                                                    inst✝² : Semiring R
                                                    n : Type u_2
                                                    inst✝¹ : Fintype n
                                                    inst✝ : DecidableEq n
                                                    I : Ideal R
                                                    M : Matrix n n R
                                                    ⊢ Iff (Membership.mem (Ideal.matricesOver n I) M) (∀ (i j : n), Membership.mem …
                                                  -/
    M ∈ I.matricesOver n ↔ ∀ i j, M i j ∈ I := by rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem matricesOver_monotone : Monotone (matricesOver (R := R) n) :=
  fun _ _ IJ _ MI i j => IJ (MI i j)


theorem matricesOver_strictMono_of_nonempty [Nonempty n] :
    StrictMono (matricesOver (R := R) n) :=
  matricesOver_monotone n |>.strictMono_of_injective <| fun I J eq => by
    /-
      R : Type u_1
      inst✝³ : Semiring R
      n : Type u_2
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : Nonempty n
      I J : Ideal R
      eq : Eq (Ideal.matricesOver n I) (Ideal.matricesOver n J)
      ⊢ Eq I J
    -/
    ext x
    /-
      case h
      R : Type u_1
      inst✝³ : Semiring R
      n : Type u_2
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : Nonempty n
      I J : Ideal R
      eq : Eq (Ideal.matricesOver n I) (Ideal.matricesOver n J)
      x : R
      ⊢ Iff (Membership.mem I x) (Membership.mem J x)
    -/
    have : (∀ _ _, x ∈ I) ↔ (∀ _ _, x ∈ J) := congr((Matrix.of fun _ _ => x) ∈ $eq)
    /-
      case h
      R : Type u_1
      inst✝³ : Semiring R
      n : Type u_2
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : Nonempty n
      I J : Ideal R
      eq : Eq (Ideal.matricesOver n I) (Ideal.matricesOver n J)
      x : R
      this : Iff (n → n → Membership.mem I x) (n → n → Membership.mem J x)
      ⊢ Iff (Membership.mem I x) (Membership.mem J x)
    -/
    simpa only [forall_const] using this
    /-
      🎉 no goals
    -/


@[simp]
theorem matricesOver_bot : (⊥ : Ideal R).matricesOver n = ⊥ := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq (Ideal.matricesOver n Bot.bot) Bot.bot
  -/
  ext M
  /-
    case h
    R : Type u_1
    inst✝² : Semiring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n R
    ⊢ Iff (Membership.mem (Ideal.matricesOver n Bot.bot) M) (Membership.mem Bot.bo …
  -/
  simp only [mem_matricesOver, mem_bot]
  /-
    case h
    R : Type u_1
    inst✝² : Semiring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n R
    ⊢ Iff (∀ (i j : n), Eq (M i j) 0) (Eq M 0)
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n R
      ⊢ (∀ (i j : n), Eq (M i j) 0) → Eq M 0
    -/
  · intro H; ext; apply H
                  /-
                    🎉 no goals
                  -/
    /-
      case h.mpr
      R : Type u_1
      inst✝² : Semiring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      M : Matrix n n R
      ⊢ Eq M 0 → ∀ (i j : n), Eq (M i j) 0
    -/
  · intro H; simp [H]
             /-
               🎉 no goals
             -/


@[simp]
theorem matricesOver_top : (⊤ : Ideal R).matricesOver n = ⊤ := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq (Ideal.matricesOver n Top.top) Top.top
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- A standard basis matrix is in $J(Mₙ(I))$
as long as its one possibly non-zero entry is in $J(I)$. -/
theorem stdBasisMatrix_mem_jacobson_matricesOver (I : Ideal R) :
    ∀ x ∈ I.jacobson, ∀ (i j : n), stdBasisMatrix i j x ∈ (I.matricesOver n).jacobson := by
  -- Proof generalized from example 8 in
  -- https://ysharifi.wordpress.com/2022/08/16/the-jacobson-radical-basic-examples/
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    ⊢ ∀ (x : R), Membership.mem I.jacobson x → ∀ (i j : n), Membership.mem (Ideal. …
  -/
  simp_rw [Ideal.mem_jacobson_iff]
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    ⊢ ∀ (x : R), (∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAd …
  -/
  intro x xIJ p q M
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    x : R
    xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
    p q : n
    M : Matrix n n R
    ⊢ Exists fun z => Membership.mem (Ideal.matricesOver n I) (HSub.hSub (HAdd.hAd …
  -/
  have ⟨z, zMx⟩ := xIJ (M q p)
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    x : R
    xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
    p q : n
    M : Matrix n n R
    z : R
    zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
    ⊢ Exists fun z => Membership.mem (Ideal.matricesOver n I) (HSub.hSub (HAdd.hAd …
  -/
  let N : Matrix n n R := 1 - ∑ i, stdBasisMatrix i q (if i = q then 1 - z else (M i p)*x*z)
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    x : R
    xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
    p q : n
    M : Matrix n n R
    z : R
    zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
    N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
    ⊢ Exists fun z => Membership.mem (Ideal.matricesOver n I) (HSub.hSub (HAdd.hAd …
  -/
  use N
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    x : R
    xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
    p q : n
    M : Matrix n n R
    z : R
    zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
    N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
    ⊢ Membership.mem (Ideal.matricesOver n I) (HSub.hSub (HAdd.hAdd (HMul.hMul (HM …
  -/
  intro i j
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    x : R
    xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
    p q : n
    M : Matrix n n R
    z : R
    zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
    N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
    i j : n
    ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul N M) (Matrix.st …
  -/
  obtain rfl | qj := eq_or_ne q j
    /-
      case h.inl
      R : Type u_1
      inst✝² : Ring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      x : R
      xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
      p q : n
      M : Matrix n n R
      z : R
      zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
      N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
      i : n
      ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul N M) (Matrix.st …
    -/
  · by_cases iq : i = q
      /-
        case pos
        R : Type u_1
        inst✝² : Ring R
        n : Type u_2
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        I : Ideal R
        x : R
        xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
        p q : n
        M : Matrix n n R
        z : R
        zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
        N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
        i : n
        iq : Eq i q
        ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul N M) (Matrix.st …
      -/
    · simp [iq, N, zMx, stdBasisMatrix, mul_apply, sum_apply, ite_and, sub_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝² : Ring R
        n : Type u_2
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        I : Ideal R
        x : R
        xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
        p q : n
        M : Matrix n n R
        z : R
        zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
        N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
        i : n
        iq : Not (Eq i q)
        ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul N M) (Matrix.st …
      -/
    · convert I.mul_mem_left (-M i p * x) zMx
      /-
        case h.e'_5
        R : Type u_1
        inst✝² : Ring R
        n : Type u_2
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        I : Ideal R
        x : R
        xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
        p q : n
        M : Matrix n n R
        z : R
        zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
        N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
        i : n
        iq : Not (Eq i q)
        ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul N M) (Matrix.stdBasisMatrix p …
      -/
      simp [iq, N, zMx, stdBasisMatrix, mul_apply, sum_apply, ite_and, sub_mul]
      /-
        case h.e'_5
        R : Type u_1
        inst✝² : Ring R
        n : Type u_2
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        I : Ideal R
        x : R
        xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
        p q : n
        M : Matrix n n R
        z : R
        zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
        N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
        i : n
        iq : Not (Eq i q)
        ⊢ Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (M i p) x) (HMul.hMul (HMul.hMul (HMul.h …
      -/
      simp [sub_add, mul_add, mul_sub, mul_assoc]
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      R : Type u_1
      inst✝² : Ring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : Ideal R
      x : R
      xIJ : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul. …
      p q : n
      M : Matrix n n R
      z : R
      zMx : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (M q p))  …
      N : Matrix n n R := HSub.hSub 1 (Finset.univ.sum fun i => Matrix.stdBasisMatri …
      i j : n
      qj : Ne q j
      ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul N M) (Matrix.st …
    -/
  · simp [N, qj, sum_apply, mul_apply]
    /-
      🎉 no goals
    -/


/-- For any left ideal $I ≤ R$, we have $Mₙ(J(I)) ≤ J(Mₙ(I))$. -/
theorem matricesOver_jacobson_le (I : Ideal R) :
    I.jacobson.matricesOver n ≤ (I.matricesOver n).jacobson := by
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    ⊢ LE.le (Ideal.matricesOver n I.jacobson) (Ideal.matricesOver n I).jacobson
  -/
  intro M MI
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    M : Matrix n n R
    MI : Membership.mem (Ideal.matricesOver n I.jacobson) M
    ⊢ Membership.mem (Ideal.matricesOver n I).jacobson M
  -/
  rw [matrix_eq_sum_stdBasisMatrix M]
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    M : Matrix n n R
    MI : Membership.mem (Ideal.matricesOver n I.jacobson) M
    ⊢ Membership.mem (Ideal.matricesOver n I).jacobson (Finset.univ.sum fun i => F …
  -/
  apply sum_mem
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    M : Matrix n n R
    MI : Membership.mem (Ideal.matricesOver n I.jacobson) M
    ⊢ ∀ (c : n), Membership.mem Finset.univ c → Membership.mem (Ideal.matricesOver …
  -/
  intro i _
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    M : Matrix n n R
    MI : Membership.mem (Ideal.matricesOver n I.jacobson) M
    i : n
    a✝ : Membership.mem Finset.univ i
    ⊢ Membership.mem (Ideal.matricesOver n I).jacobson (Finset.univ.sum fun j => M …
  -/
  apply sum_mem
  /-
    case h.h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    M : Matrix n n R
    MI : Membership.mem (Ideal.matricesOver n I.jacobson) M
    i : n
    a✝ : Membership.mem Finset.univ i
    ⊢ ∀ (c : n), Membership.mem Finset.univ c → Membership.mem (Ideal.matricesOver …
  -/
  intro j _
  /-
    case h.h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : Ideal R
    M : Matrix n n R
    MI : Membership.mem (Ideal.matricesOver n I.jacobson) M
    i : n
    a✝¹ : Membership.mem Finset.univ i
    j : n
    a✝ : Membership.mem Finset.univ j
    ⊢ Membership.mem (Ideal.matricesOver n I).jacobson (Matrix.stdBasisMatrix i j  …
  -/
  apply stdBasisMatrix_mem_jacobson_matricesOver I _ (MI i j)
  /-
    🎉 no goals
  -/


/-- The two-sided ideal of matrices with entries in `I ≤ R`. -/
def matricesOver (I : TwoSidedIdeal R) : TwoSidedIdeal (Matrix n n R) :=
  TwoSidedIdeal.mk' { M | ∀ i j, M i j ∈ I }
    (fun _ _ => I.zero_mem)
    (fun ha hb i j => I.add_mem (ha i j) (hb i j))
    (fun ha i j => I.neg_mem (ha i j))
    (fun ha i j => by
      /-
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) y✝
        i j : n
        ⊢ Membership.mem I (HMul.hMul x✝ y✝ i j)
      -/
      rw [mul_apply]
      /-
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) y✝
        i j : n
        ⊢ Membership.mem I (Finset.univ.sum fun j_1 => HMul.hMul (x✝ i j_1) (y✝ j_1 j))
      -/
      apply sum_mem
      /-
        case h
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) y✝
        i j : n
        ⊢ ∀ (c : n), Membership.mem Finset.univ c → Membership.mem I (HMul.hMul (x✝ i  …
      -/
      intro k _
      /-
        case h
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) y✝
        i j k : n
        a✝ : Membership.mem Finset.univ k
        ⊢ Membership.mem I (HMul.hMul (x✝ i k) (y✝ k j))
      -/
      apply I.mul_mem_left _ _ (ha k j))
      /-
        🎉 no goals
      -/
    (fun ha i j => by
      /-
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) x✝
        i j : n
        ⊢ Membership.mem I (HMul.hMul x✝ y✝ i j)
      -/
      rw [mul_apply]
      /-
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) x✝
        i j : n
        ⊢ Membership.mem I (Finset.univ.sum fun j_1 => HMul.hMul (x✝ i j_1) (y✝ j_1 j))
      -/
      apply sum_mem
      /-
        case h
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) x✝
        i j : n
        ⊢ ∀ (c : n), Membership.mem Finset.univ c → Membership.mem I (HMul.hMul (x✝ i  …
      -/
      intro k _
      /-
        case h
        R : Type u_1
        inst✝¹ : Ring R
        n : Type u_2
        inst✝ : Fintype n
        I : TwoSidedIdeal R
        x✝ y✝ : Matrix n n R
        ha : Membership.mem (setOf fun M => ∀ (i j : n), Membership.mem I (M i j)) x✝
        i j k : n
        a✝ : Membership.mem Finset.univ k
        ⊢ Membership.mem I (HMul.hMul (x✝ i k) (y✝ k j))
      -/
      apply I.mul_mem_right _ _ (ha i k))
      /-
        🎉 no goals
      -/


@[simp]
lemma mem_matricesOver (I : TwoSidedIdeal R) (M : Matrix n n R) :
    M ∈ I.matricesOver n ↔ ∀ i j, M i j ∈ I := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    n : Type u_2
    inst✝ : Fintype n
    I : TwoSidedIdeal R
    M : Matrix n n R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.matricesOver n I) M) (∀ (i j : n), Member …
  -/
  simp [matricesOver]
  /-
    🎉 no goals
  -/


theorem matricesOver_strictMono_of_nonempty [h : Nonempty n] :
    StrictMono (matricesOver (R := R) n) :=
  matricesOver_monotone n |>.strictMono_of_injective <| fun I J eq => by
    /-
      R : Type u_1
      inst✝¹ : Ring R
      n : Type u_2
      inst✝ : Fintype n
      h : Nonempty n
      I J : TwoSidedIdeal R
      eq : Eq (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOver n J)
      ⊢ Eq I J
    -/
    ext x
    /-
      case h
      R : Type u_1
      inst✝¹ : Ring R
      n : Type u_2
      inst✝ : Fintype n
      h : Nonempty n
      I J : TwoSidedIdeal R
      eq : Eq (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOver n J)
      x : R
      ⊢ Iff (Membership.mem I x) (Membership.mem J x)
    -/
    have : _ ↔ _ := congr((Matrix.of fun _ _ => x) ∈ $eq)
    /-
      case h
      R : Type u_1
      inst✝¹ : Ring R
      n : Type u_2
      inst✝ : Fintype n
      h : Nonempty n
      I J : TwoSidedIdeal R
      eq : Eq (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOver n J)
      x : R
      this : Iff (Membership.mem (Mathlib.Tactic.TermCongr.cHole (TwoSidedIdeal.matr …
      ⊢ Iff (Membership.mem I x) (Membership.mem J x)
    -/
    simpa only [mem_matricesOver, of_apply, forall_const] using this
    /-
      🎉 no goals
    -/


@[simp]
theorem matricesOver_bot : (⊥ : TwoSidedIdeal R).matricesOver n = ⊥ := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    n : Type u_2
    inst✝ : Fintype n
    ⊢ Eq (TwoSidedIdeal.matricesOver n Bot.bot) Bot.bot
  -/
  ext M
  /-
    case h
    R : Type u_1
    inst✝¹ : Ring R
    n : Type u_2
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.matricesOver n Bot.bot) M) (Membership.me …
  -/
  simp only [mem_matricesOver, mem_bot]
  /-
    case h
    R : Type u_1
    inst✝¹ : Ring R
    n : Type u_2
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Iff (∀ (i j : n), Eq (M i j) 0) (Eq M 0)
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      inst✝¹ : Ring R
      n : Type u_2
      inst✝ : Fintype n
      M : Matrix n n R
      ⊢ (∀ (i j : n), Eq (M i j) 0) → Eq M 0
    -/
  · intro H; ext; apply H
                  /-
                    🎉 no goals
                  -/
    /-
      case h.mpr
      R : Type u_1
      inst✝¹ : Ring R
      n : Type u_2
      inst✝ : Fintype n
      M : Matrix n n R
      ⊢ Eq M 0 → ∀ (i j : n), Eq (M i j) 0
    -/
  · intro H; simp [H]
             /-
               🎉 no goals
             -/


@[simp]
theorem matricesOver_top : (⊤ : TwoSidedIdeal R).matricesOver n = ⊤ := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    n : Type u_2
    inst✝ : Fintype n
    ⊢ Eq (TwoSidedIdeal.matricesOver n Top.top) Top.top
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem asIdeal_matricesOver [DecidableEq n] (I : TwoSidedIdeal R) :
    asIdeal (I.matricesOver n) = (asIdeal I).matricesOver n := by
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    ⊢ Eq (TwoSidedIdeal.asIdeal (TwoSidedIdeal.matricesOver n I)) (Ideal.matricesO …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/--
Two-sided ideals in $R$ correspond bijectively to those in $Mₙ(R)$.
Given an ideal $I ≤ R$, we send it to $Mₙ(I)$.
Given an ideal $J ≤ Mₙ(R)$, we send it to $\{Nᵢⱼ ∣ ∃ N ∈ J\}$.
-/
@[simps]
def equivMatricesOver (i j : n) : TwoSidedIdeal R ≃ TwoSidedIdeal (Matrix n n R) where
  toFun I := I.matricesOver n
  invFun J := TwoSidedIdeal.mk'
    { N i j | N ∈ J }
    ⟨0, J.zero_mem, rfl⟩
        /-
          R : Type u_1
          inst✝² : Ring R
          n✝ : Type u_2
          inst✝¹ : Fintype n✝
          n : Type u_3
          inst✝ : Fintype n
          i j : n
          J : TwoSidedIdeal (Matrix n n R)
          ⊢ ∀ {x y : R}, Membership.mem (setOf fun x => Exists fun N => And (Membership. …
        -/
    (by rintro _ _ ⟨x, hx, rfl⟩ ⟨y, hy, rfl⟩; exact ⟨x + y, J.add_mem hx hy, rfl⟩)
                                              /-
                                                🎉 no goals
                                              -/
        /-
          R : Type u_1
          inst✝² : Ring R
          n✝ : Type u_2
          inst✝¹ : Fintype n✝
          n : Type u_3
          inst✝ : Fintype n
          i j : n
          J : TwoSidedIdeal (Matrix n n R)
          ⊢ ∀ {x : R}, Membership.mem (setOf fun x => Exists fun N => And (Membership.me …
        -/
    (by rintro _ ⟨x, hx, rfl⟩; exact ⟨-x, J.neg_mem hx, rfl⟩)
                               /-
                                 🎉 no goals
                               -/
    (by
      classical
      rintro x _ ⟨y, hy, rfl⟩
      exact ⟨diagonal (fun _ ↦ x) * y, J.mul_mem_left _ _ hy, by simp⟩)
    (by
      classical
      rintro _ y ⟨x, hx, rfl⟩
      exact ⟨x * diagonal (fun _ ↦ y), J.mul_mem_right _ _ hx, by simp⟩)
  right_inv J := SetLike.ext fun x ↦ by
    classical
    simp only [mem_mk', Set.mem_image, SetLike.mem_coe, mem_matricesOver]
    constructor
    · intro h
      choose y hy1 hy2 using h
      rw [matrix_eq_sum_stdBasisMatrix x]
      refine sum_mem fun k _ ↦ sum_mem fun l _ ↦ ?_
      suffices
          stdBasisMatrix k l (x k l) =
          stdBasisMatrix k i 1 * y k l * stdBasisMatrix j l 1 by
        rw [this]
        exact J.mul_mem_right _ _ (J.mul_mem_left _ _ <| hy1 _ _)
      ext a b
      by_cases hab : a = k ∧ b = l
      · rcases hab with ⟨ha, hb⟩
        subst ha hb
        simp only [StdBasisMatrix.apply_same, StdBasisMatrix.mul_right_apply_same,
          StdBasisMatrix.mul_left_apply_same, one_mul, mul_one]
        rw [hy2 a b]
      · conv_lhs =>
          dsimp [stdBasisMatrix]
          rw [if_neg (by tauto)]
        rw [not_and_or] at hab
        rcases hab with ha | hb
        · rw [mul_assoc, StdBasisMatrix.mul_left_apply_of_ne (h := ha)]
        · rw [StdBasisMatrix.mul_right_apply_of_ne (hbj := hb)]
    · intro hx k l
      refine ⟨stdBasisMatrix i k 1 * x * stdBasisMatrix l j 1,
        J.mul_mem_right _ _ (J.mul_mem_left _ _ hx), ?_⟩
      rw [StdBasisMatrix.mul_right_apply_same, StdBasisMatrix.mul_left_apply_same,
        mul_one, one_mul]
  left_inv I := SetLike.ext fun x ↦ by
    /-
      R : Type u_1
      inst✝² : Ring R
      n✝ : Type u_2
      inst✝¹ : Fintype n✝
      n : Type u_3
      inst✝ : Fintype n
      i j : n
      I : TwoSidedIdeal R
      x : R
      ⊢ Iff (Membership.mem ((fun J => TwoSidedIdeal.mk' (setOf fun x => Exists fun  …
    -/
    simp only [mem_mk', Set.mem_image, SetLike.mem_coe, mem_matricesOver]
    /-
      R : Type u_1
      inst✝² : Ring R
      n✝ : Type u_2
      inst✝¹ : Fintype n✝
      n : Type u_3
      inst✝ : Fintype n
      i j : n
      I : TwoSidedIdeal R
      x : R
      ⊢ Iff (Membership.mem (setOf fun x => Exists fun N => And (∀ (i j : n), Member …
    -/
    constructor
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I : TwoSidedIdeal R
        x : R
        ⊢ Membership.mem (setOf fun x => Exists fun N => And (∀ (i j : n), Membership. …
      -/
    · intro h
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I : TwoSidedIdeal R
        x : R
        h : Membership.mem (setOf fun x => Exists fun N => And (∀ (i j : n), Membershi …
        ⊢ Membership.mem I x
      -/
      choose y hy1 hy2 using h
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I : TwoSidedIdeal R
        x : R
        y : Matrix n n R
        hy1 : ∀ (i j : n), Membership.mem I (y i j)
        hy2 : Eq (y i j) x
        ⊢ Membership.mem I x
      -/
      exact hy2 ▸ hy1 _ _
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I : TwoSidedIdeal R
        x : R
        ⊢ Membership.mem I x → Membership.mem (setOf fun x => Exists fun N => And (∀ ( …
      -/
    · intro h
      /-
        case mpr
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I : TwoSidedIdeal R
        x : R
        h : Membership.mem I x
        ⊢ Membership.mem (setOf fun x => Exists fun N => And (∀ (i j : n), Membership. …
      -/
      exact ⟨of fun _ _ => x, by simp [h], rfl⟩
      /-
        🎉 no goals
      -/


/--
Two-sided ideals in $R$ are order-isomorphic with those in $Mₙ(R)$.
See also `equivMatricesOver`.
-/
@[simps!]
def orderIsoMatricesOver (i j : n) : TwoSidedIdeal R ≃o TwoSidedIdeal (Matrix n n R) where
  __ := equivMatricesOver i j
  map_rel_iff' {I J} := by
    /-
      R : Type u_1
      inst✝² : Ring R
      n✝ : Type u_2
      inst✝¹ : Fintype n✝
      n : Type u_3
      inst✝ : Fintype n
      i j : n
      I J : TwoSidedIdeal R
      ⊢ Iff (LE.le (__spread✝⁻⁰ I) (__spread✝⁻⁰ J)) (LE.le I J)
    -/
    simp only [equivMatricesOver_apply]
    /-
      R : Type u_1
      inst✝² : Ring R
      n✝ : Type u_2
      inst✝¹ : Fintype n✝
      n : Type u_3
      inst✝ : Fintype n
      i j : n
      I J : TwoSidedIdeal R
      ⊢ Iff (LE.le (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOver n J) …
    -/
    constructor
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I J : TwoSidedIdeal R
        ⊢ LE.le (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOver n J) → LE …
      -/
    · intro le x xI
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I J : TwoSidedIdeal R
        le : LE.le (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOver n J)
        x : R
        xI : Membership.mem I x
        ⊢ Membership.mem J x
      -/
      specialize @le (of fun _ _ => x) (by simp [xI])
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I J : TwoSidedIdeal R
        x : R
        xI : Membership.mem I x
        le : Membership.mem (TwoSidedIdeal.matricesOver n J) (Matrix.of fun x_1 x_2 => …
        ⊢ Membership.mem J x
      -/
      letI : Inhabited n := ⟨i⟩
      /-
        case mp
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I J : TwoSidedIdeal R
        x : R
        xI : Membership.mem I x
        le : Membership.mem (TwoSidedIdeal.matricesOver n J) (Matrix.of fun x_1 x_2 => …
        this : Inhabited n := { default := i }
        ⊢ Membership.mem J x
      -/
      simpa using le
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i j : n
        I J : TwoSidedIdeal R
        ⊢ LE.le I J → LE.le (TwoSidedIdeal.matricesOver n I) (TwoSidedIdeal.matricesOv …
      -/
    · intro IJ M MI i j
      /-
        case mpr
        R : Type u_1
        inst✝² : Ring R
        n✝ : Type u_2
        inst✝¹ : Fintype n✝
        n : Type u_3
        inst✝ : Fintype n
        i✝ j✝ : n
        I J : TwoSidedIdeal R
        IJ : LE.le I J
        M : Matrix n n R
        MI : Membership.mem (TwoSidedIdeal.matricesOver n I) M
        i j : n
        ⊢ Membership.mem J (HSub.hSub M 0 i j)
      -/
      exact IJ <| MI i j
      /-
        🎉 no goals
      -/


private lemma jacobson_matricesOver_le (I : TwoSidedIdeal R) :
    (I.matricesOver n).jacobson ≤ I.jacobson.matricesOver n := by
  -- Proof generalized from example 8 in
  -- https://ysharifi.wordpress.com/2022/08/16/the-jacobson-radical-basic-examples/
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    ⊢ LE.le (TwoSidedIdeal.matricesOver n I).jacobson (TwoSidedIdeal.matricesOver  …
  -/
  intro M Mmem p q
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    Mmem : Membership.mem (TwoSidedIdeal.matricesOver n I).jacobson M
    p q : n
    ⊢ Membership.mem I.jacobson (HSub.hSub M 0 p q)
  -/
  rw [sub_zero, mem_jacobson_iff]
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    Mmem : Membership.mem (TwoSidedIdeal.matricesOver n I).jacobson M
    p q : n
    ⊢ ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul …
  -/
  replace Mmem := mul_mem_right _ _ (stdBasisMatrix q p 1) Mmem
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    p q : n
    Mmem : Membership.mem (TwoSidedIdeal.matricesOver n I).jacobson (HMul.hMul M ( …
    ⊢ ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul …
  -/
  rw [mem_jacobson_iff] at Mmem
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    p q : n
    Mmem : ∀ (y : Matrix n n R), Exists fun z => Membership.mem (TwoSidedIdeal.mat …
    ⊢ ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul …
  -/
  intro y
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    p q : n
    Mmem : ∀ (y : Matrix n n R), Exists fun z => Membership.mem (TwoSidedIdeal.mat …
    y : R
    ⊢ Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul …
  -/
  specialize Mmem (y • stdBasisMatrix p p 1)
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    p q : n
    y : R
    Mmem : Exists fun z => Membership.mem (TwoSidedIdeal.matricesOver n I) (HSub.h …
    ⊢ Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul …
  -/
  have ⟨N, NxMI⟩ := Mmem
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    p q : n
    y : R
    Mmem : Exists fun z => Membership.mem (TwoSidedIdeal.matricesOver n I) (HSub.h …
    N : Matrix n n R
    NxMI : Membership.mem (TwoSidedIdeal.matricesOver n I) (HSub.hSub (HAdd.hAdd ( …
    ⊢ Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul …
  -/
  use N p p
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    M : Matrix n n R
    p q : n
    y : R
    Mmem : Exists fun z => Membership.mem (TwoSidedIdeal.matricesOver n I) (HSub.h …
    N : Matrix n n R
    NxMI : Membership.mem (TwoSidedIdeal.matricesOver n I) (HSub.hSub (HAdd.hAdd ( …
    ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul (N p p) y) (M p …
  -/
  simpa [mul_apply, stdBasisMatrix, ite_and] using NxMI p p
  /-
    🎉 no goals
  -/


/-- For any two-sided ideal $I ≤ R$, we have $J(Mₙ(I)) = Mₙ(J(I))$. -/
theorem jacobson_matricesOver (I : TwoSidedIdeal R) :
    (I.matricesOver n).jacobson = I.jacobson.matricesOver n := by
  /-
    R : Type u_1
    inst✝² : Ring R
    n : Type u_2
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    I : TwoSidedIdeal R
    ⊢ Eq (TwoSidedIdeal.matricesOver n I).jacobson (TwoSidedIdeal.matricesOver n I …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝² : Ring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : TwoSidedIdeal R
      ⊢ LE.le (TwoSidedIdeal.matricesOver n I).jacobson (TwoSidedIdeal.matricesOver  …
    -/
  · apply jacobson_matricesOver_le
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      inst✝² : Ring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : TwoSidedIdeal R
      ⊢ LE.le (TwoSidedIdeal.matricesOver n I.jacobson) (TwoSidedIdeal.matricesOver  …
    -/
  · show asIdeal (I.matricesOver n).jacobson ≥ asIdeal (I.jacobson.matricesOver n)
    /-
      case a
      R : Type u_1
      inst✝² : Ring R
      n : Type u_2
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      I : TwoSidedIdeal R
      ⊢ GE.ge (TwoSidedIdeal.asIdeal (TwoSidedIdeal.matricesOver n I).jacobson) (Two …
    -/
    simp [asIdeal_jacobson, asIdeal_matricesOver, Ideal.matricesOver_jacobson_le]
    /-
      🎉 no goals
    -/


theorem matricesOver_jacobson_bot :
    (⊥ : TwoSidedIdeal R).jacobson.matricesOver n = (⊥ : TwoSidedIdeal (Matrix n n R)).jacobson :=
  matricesOver_bot n (R := R) ▸ (jacobson_matricesOver _).symm


