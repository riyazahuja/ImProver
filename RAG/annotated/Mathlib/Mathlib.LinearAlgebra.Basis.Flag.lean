/-- The subspace spanned by the first `k` vectors of the basis `b`. -/
def flag (b : Basis (Fin n) R M) (k : Fin (n + 1)) : Submodule R M :=
  .span R <| b '' {i | i.castSucc < k}


@[simp]
                                                               /-
                                                                 R : Type u_1
                                                                 M : Type u_2
                                                                 inst✝² : Semiring R
                                                                 inst✝¹ : AddCommMonoid M
                                                                 inst✝ : Module R M
                                                                 n : Nat
                                                                 b : Basis (Fin n) R M
                                                                 ⊢ Eq (b.flag 0) Bot.bot
                                                               -/
theorem flag_zero (b : Basis (Fin n) R M) : b.flag 0 = ⊥ := by simp [flag]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem flag_last (b : Basis (Fin n) R M) : b.flag (.last n) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    ⊢ Eq (b.flag (Fin.last n)) Top.top
  -/
  simp [flag, Fin.castSucc_lt_last]
  /-
    🎉 no goals
  -/


theorem flag_le_iff (b : Basis (Fin n) R M) {k p} :
    b.flag k ≤ p ↔ ∀ i : Fin n, i.castSucc < k → b i ∈ p :=
  span_le.trans forall_mem_image


theorem flag_succ (b : Basis (Fin n) R M) (k : Fin n) :
    b.flag k.succ = (R ∙ b k) ⊔ b.flag k.castSucc := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    k : Fin n
    ⊢ Eq (b.flag k.succ) (Max.max (Submodule.span R (Singleton.singleton (b k))) ( …
  -/
  simp only [flag, Fin.castSucc_lt_castSucc_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    k : Fin n
    ⊢ Eq (Submodule.span R (Set.image (⇑b) (setOf fun i => LT.lt i.castSucc k.succ …
  -/
  simp [Fin.castSucc_lt_iff_succ_le, le_iff_eq_or_lt, setOf_or, image_insert_eq, span_insert]
  /-
    🎉 no goals
  -/


theorem self_mem_flag (b : Basis (Fin n) R M) {i : Fin n} {k : Fin (n + 1)} (h : i.castSucc < k) :
    b i ∈ b.flag k :=
  subset_span <| mem_image_of_mem _ h


@[simp]
theorem self_mem_flag_iff [Nontrivial R] (b : Basis (Fin n) R M) {i : Fin n} {k : Fin (n + 1)} :
    b i ∈ b.flag k ↔ i.castSucc < k :=
  b.self_mem_span_image


@[mono]
theorem flag_mono (b : Basis (Fin n) R M) : Monotone b.flag :=
                                        /-
                                          R : Type u_1
                                          M : Type u_2
                                          inst✝² : Semiring R
                                          inst✝¹ : AddCommMonoid M
                                          inst✝ : Module R M
                                          n : Nat
                                          b : Basis (Fin n) R M
                                          k : Fin n
                                          ⊢ LE.le (b.flag k.castSucc) (b.flag k.succ)
                                        -/
  Fin.monotone_iff_le_succ.2 fun k ↦ by rw [flag_succ]; exact le_sup_right
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem isChain_range_flag (b : Basis (Fin n) R M) : IsChain (· ≤ ·) (range b.flag) :=
  b.flag_mono.isChain_range


@[mono]
theorem flag_strictMono [Nontrivial R] (b : Basis (Fin n) R M) : StrictMono b.flag :=
                                          /-
                                            R : Type u_1
                                            M : Type u_2
                                            inst✝³ : Semiring R
                                            inst✝² : AddCommMonoid M
                                            inst✝¹ : Module R M
                                            n : Nat
                                            inst✝ : Nontrivial R
                                            b : Basis (Fin n) R M
                                            x✝ : Fin n
                                            ⊢ LT.lt (b.flag x✝.castSucc) (b.flag x✝.succ)
                                          -/
  Fin.strictMono_iff_lt_succ.2 fun _ ↦ by simp [flag_succ]
                                          /-
                                            🎉 no goals
                                          -/


@[gcongr] lemma flag_le_flag (hij : i ≤ j) : b.flag i ≤ b.flag j := flag_mono _ hij


@[gcongr]
lemma flag_lt_flag [Nontrivial R] (hij : i < j) : b.flag i < b.flag j := flag_strictMono _ hij


@[simp]
theorem flag_le_ker_coord_iff [Nontrivial R] (b : Basis (Fin n) R M) {k : Fin (n + 1)} {l : Fin n} :
    b.flag k ≤ LinearMap.ker (b.coord l) ↔ k ≤ l.castSucc := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    n : Nat
    inst✝ : Nontrivial R
    b : Basis (Fin n) R M
    k : Fin (HAdd.hAdd n 1)
    l : Fin n
    ⊢ Iff (LE.le (b.flag k) (LinearMap.ker (b.coord l))) (LE.le k l.castSucc)
  -/
  simp [flag_le_iff, Finsupp.single_apply_eq_zero, imp_false, imp_not_comm]
  /-
    🎉 no goals
  -/


theorem flag_le_ker_coord (b : Basis (Fin n) R M) {k : Fin (n + 1)} {l : Fin n}
    (h : k ≤ l.castSucc) : b.flag k ≤ LinearMap.ker (b.coord l) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    k : Fin (HAdd.hAdd n 1)
    l : Fin n
    h : LE.le k l.castSucc
    ⊢ LE.le (b.flag k) (LinearMap.ker (b.coord l))
  -/
  nontriviality R
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    k : Fin (HAdd.hAdd n 1)
    l : Fin n
    h : LE.le k l.castSucc
    a✝ : Nontrivial R
    ⊢ LE.le (b.flag k) (LinearMap.ker (b.coord l))
  -/
  exact b.flag_le_ker_coord_iff.2 h
  /-
    🎉 no goals
  -/


theorem flag_le_ker_dual (b : Basis (Fin n) R M) (k : Fin n) :
    b.flag k.castSucc ≤ LinearMap.ker (b.dualBasis k) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    k : Fin n
    ⊢ LE.le (b.flag k.castSucc) (LinearMap.ker (b.dualBasis k))
  -/
  nontriviality R
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    b : Basis (Fin n) R M
    k : Fin n
    a✝ : Nontrivial R
    ⊢ LE.le (b.flag k.castSucc) (LinearMap.ker (b.dualBasis k))
  -/
  rw [coe_dualBasis, b.flag_le_ker_coord_iff]
  /-
    🎉 no goals
  -/


theorem flag_covBy (b : Basis (Fin n) K V) (i : Fin n) :
    b.flag i.castSucc ⋖ b.flag i.succ := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    n : Nat
    b : Basis (Fin n) K V
    i : Fin n
    ⊢ CovBy (b.flag i.castSucc) (b.flag i.succ)
  -/
  rw [flag_succ]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    n : Nat
    b : Basis (Fin n) K V
    i : Fin n
    ⊢ CovBy (b.flag i.castSucc) (Max.max (Submodule.span K (Singleton.singleton (b …
  -/
  apply covBy_span_singleton_sup
  /-
    case h
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    n : Nat
    b : Basis (Fin n) K V
    i : Fin n
    ⊢ Not (Membership.mem (b.flag i.castSucc) (b i))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem flag_wcovBy (b : Basis (Fin n) K V) (i : Fin n) :
    b.flag i.castSucc ⩿ b.flag i.succ :=
  (b.flag_covBy i).wcovBy


/-- Range of `Basis.flag` as a `Flag`. -/
@[simps!]
def toFlag (b : Basis (Fin n) K V) : Flag (Submodule K V) :=
  .rangeFin b.flag b.flag_zero b.flag_last b.flag_wcovBy


@[simp]
theorem mem_toFlag (b : Basis (Fin n) K V) {p : Submodule K V} : p ∈ b.toFlag ↔ ∃ k, b.flag k = p :=
  Iff.rfl


theorem isMaxChain_range_flag (b : Basis (Fin n) K V) : IsMaxChain (· ≤ ·) (range b.flag) :=
  b.toFlag.maxChain


