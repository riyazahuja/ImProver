/-- The submodule `genEigenspace f μ k` for a linear map `f`, a scalar `μ`,
and a number `k : ℕ∞` is the kernel of `(f - μ • id) ^ k` if `k` is a natural number
(see Def 8.10 of [axler2015]), or the union of all these kernels if `k = ∞`.
A generalized eigenspace for some exponent `k` is contained in
the generalized eigenspace for exponents larger than `k`. -/
def genEigenspace (f : End R M) (μ : R) : ℕ∞ →o Submodule R M where
  toFun k := ⨆ l : ℕ, ⨆ _ : l ≤ k, LinearMap.ker ((f - μ • 1) ^ l)
  monotone' _ _ hkl := biSup_mono fun _ hi ↦ hi.trans hkl


lemma mem_genEigenspace {f : End R M} {μ : R} {k : ℕ∞} {x : M} :
    x ∈ f.genEigenspace μ k ↔ ∃ l : ℕ, l ≤ k ∧ x ∈ LinearMap.ker ((f - μ • 1) ^ l) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : ENat
    x : M
    ⊢ Iff (Membership.mem ((f.genEigenspace μ) k) x) (Exists fun l => And (LE.le ( …
  -/
  have : Nonempty {l : ℕ // l ≤ k} := ⟨⟨0, zero_le _⟩⟩
  have : Directed (ι := { i : ℕ // i ≤ k }) (· ≤ ·) fun i ↦ LinearMap.ker ((f - μ • 1) ^ (i : ℕ)) :=
    Monotone.directed_le fun m n h ↦ by simpa using (f - μ • 1).iterateKer.monotone h
  simp_rw [genEigenspace, OrderHom.coe_mk, LinearMap.mem_ker, iSup_subtype',
    Submodule.mem_iSup_of_directed _ this, LinearMap.mem_ker, Subtype.exists, exists_prop]


lemma genEigenspace_directed {f : End R M} {μ : R} {k : ℕ∞} :
    Directed (· ≤ ·) (fun l : {l : ℕ // l ≤ k} ↦ f.genEigenspace μ l) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : ENat
    ⊢ Directed (fun x1 x2 => LE.le x1 x2) fun l => (f.genEigenspace μ) ↑↑l
  -/
  have aux : Monotone ((↑) : {l : ℕ // l ≤ k} → ℕ∞) := fun x y h ↦ by simpa using h
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : ENat
    aux : Monotone fun x => ↑↑x
    ⊢ Directed (fun x1 x2 => LE.le x1 x2) fun l => (f.genEigenspace μ) ↑↑l
  -/
  exact ((genEigenspace f μ).monotone.comp aux).directed_le
  /-
    🎉 no goals
  -/


lemma mem_genEigenspace_nat {f : End R M} {μ : R} {k : ℕ} {x : M} :
    x ∈ f.genEigenspace μ k ↔ x ∈ LinearMap.ker ((f - μ • 1) ^ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    x : M
    ⊢ Iff (Membership.mem ((f.genEigenspace μ) ↑k) x) (Membership.mem (LinearMap.k …
  -/
  rw [mem_genEigenspace]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    x : M
    ⊢ Iff (Exists fun l => And (LE.le ↑l ↑k) (Membership.mem (LinearMap.ker (HPow. …
  -/
  constructor
    /-
      case mp
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      ⊢ (Exists fun l => And (LE.le ↑l ↑k) (Membership.mem (LinearMap.ker (HPow.hPow …
    -/
  · rintro ⟨l, hl, hx⟩
    /-
      case mp.intro.intro
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      l : Nat
      hl : LE.le ↑l ↑k
      hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
      ⊢ Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k)) x
    -/
    simp only [Nat.cast_le] at hl
    /-
      case mp.intro.intro
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      l : Nat
      hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
      hl : LE.le l k
      ⊢ Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k)) x
    -/
    exact (f - μ • 1).iterateKer.monotone hl hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      ⊢ Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k)) …
    -/
  · intro hx
    /-
      case mpr
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
      ⊢ Exists fun l => And (LE.le ↑l ↑k) (Membership.mem (LinearMap.ker (HPow.hPow  …
    -/
    exact ⟨k, le_rfl, hx⟩
    /-
      🎉 no goals
    -/


lemma mem_genEigenspace_top {f : End R M} {μ : R} {x : M} :
    x ∈ f.genEigenspace μ ⊤ ↔ ∃ k : ℕ, x ∈ LinearMap.ker ((f - μ • 1) ^ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    x : M
    ⊢ Iff (Membership.mem ((f.genEigenspace μ) Top.top) x) (Exists fun k => Member …
  -/
  simp [mem_genEigenspace]
  /-
    🎉 no goals
  -/


lemma genEigenspace_nat {f : End R M} {μ : R} {k : ℕ} :
    f.genEigenspace μ k = LinearMap.ker ((f - μ • 1) ^ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    ⊢ Eq ((f.genEigenspace μ) ↑k) (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hS …
  -/
  ext; simp [mem_genEigenspace_nat]
       /-
         🎉 no goals
       -/


lemma genEigenspace_eq_iSup_genEigenspace_nat (f : End R M) (μ : R) (k : ℕ∞) :
    f.genEigenspace μ k = ⨆ l : {l : ℕ // l ≤ k}, f.genEigenspace μ l := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : ENat
    ⊢ Eq ((f.genEigenspace μ) k) (iSup fun l => (f.genEigenspace μ) ↑↑l)
  -/
  simp_rw [genEigenspace_nat, genEigenspace, OrderHom.coe_mk, iSup_subtype]
  /-
    🎉 no goals
  -/


lemma genEigenspace_top (f : End R M) (μ : R) :
    f.genEigenspace μ ⊤ = ⨆ k : ℕ, f.genEigenspace μ k := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ Eq ((f.genEigenspace μ) Top.top) (iSup fun k => (f.genEigenspace μ) ↑k)
  -/
  rw [genEigenspace_eq_iSup_genEigenspace_nat, iSup_subtype]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ Eq (iSup fun i => iSup fun h => (f.genEigenspace μ) ↑↑⟨i, h⟩) (iSup fun k => …
  -/
  simp only [le_top, iSup_pos, OrderHom.coe_mk]
  /-
    🎉 no goals
  -/


lemma genEigenspace_one {f : End R M} {μ : R} :
    f.genEigenspace μ 1 = LinearMap.ker (f - μ • 1) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ Eq ((f.genEigenspace μ) 1) (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1)))
  -/
  rw [← Nat.cast_one, genEigenspace_nat, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma mem_genEigenspace_one {f : End R M} {μ : R} {x : M} :
    x ∈ f.genEigenspace μ 1 ↔ f x = μ • x := by
  rw [genEigenspace_one, LinearMap.mem_ker, LinearMap.sub_apply,
    sub_eq_zero, LinearMap.smul_apply, LinearMap.one_apply]

-- `simp` can prove this using `genEigenspace_zero`

lemma mem_genEigenspace_zero {f : End R M} {μ : R} {x : M} :
    x ∈ f.genEigenspace μ 0 ↔ x = 0 := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    x : M
    ⊢ Iff (Membership.mem ((f.genEigenspace μ) 0) x) (Eq x 0)
  -/
  rw [← Nat.cast_zero, mem_genEigenspace_nat, pow_zero, LinearMap.mem_ker, LinearMap.one_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma genEigenspace_zero {f : End R M} {μ : R} :
    f.genEigenspace μ 0 = ⊥ := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ Eq ((f.genEigenspace μ) 0) Bot.bot
  -/
  ext; apply mem_genEigenspace_zero
       /-
         🎉 no goals
       -/


@[simp]
lemma genEigenspace_zero_nat (f : End R M) (k : ℕ) :
    f.genEigenspace 0 k = LinearMap.ker (f ^ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    k : Nat
    ⊢ Eq ((f.genEigenspace 0) ↑k) (LinearMap.ker (HPow.hPow f k))
  -/
  ext; simp [mem_genEigenspace_nat]
       /-
         🎉 no goals
       -/


/-- Let `M` be an `R`-module, and `f` an `R`-linear endomorphism of `M`,
and let `μ : R` and `k : ℕ∞` be given.
Then `x : M` satisfies `HasUnifEigenvector f μ k x` if
`x ∈ f.genEigenspace μ k` and `x ≠ 0`.

For `k = 1`, this means that `x` is an eigenvector of `f` with eigenvalue `μ`. -/
def HasUnifEigenvector (f : End R M) (μ : R) (k : ℕ∞) (x : M) : Prop :=
  x ∈ f.genEigenspace μ k ∧ x ≠ 0


/-- Let `M` be an `R`-module, and `f` an `R`-linear endomorphism of `M`.
Then `μ : R` and `k : ℕ∞` satisfy `HasUnifEigenvalue f μ k` if
`f.genEigenspace μ k ≠ ⊥`.

For `k = 1`, this means that `μ` is an eigenvalue of `f`. -/
def HasUnifEigenvalue (f : End R M) (μ : R) (k : ℕ∞) : Prop :=
  f.genEigenspace μ k ≠ ⊥


/-- Let `M` be an `R`-module, and `f` an `R`-linear endomorphism of `M`.
For `k : ℕ∞`, we define `UnifEigenvalues f k` to be the type of all
`μ : R` that satisfy `f.HasUnifEigenvalue μ k`.

For `k = 1` this is the type of all eigenvalues of `f`. -/
def UnifEigenvalues (f : End R M) (k : ℕ∞) : Type _ :=
  { μ : R // f.HasUnifEigenvalue μ k }


/-- The underlying value of a bundled eigenvalue. -/
@[coe]
def UnifEigenvalues.val (f : Module.End R M) (k : ℕ∞) : UnifEigenvalues f k → R := Subtype.val


instance UnifEigenvalues.instCoeOut {f : Module.End R M} (k : ℕ∞) :
    CoeOut (UnifEigenvalues f k) R where
  coe := UnifEigenvalues.val f k


instance UnivEigenvalues.instDecidableEq [DecidableEq R] (f : Module.End R M) (k : ℕ∞) :
    DecidableEq (UnifEigenvalues f k) :=
  inferInstanceAs (DecidableEq (Subtype (fun x : R ↦ f.HasUnifEigenvalue x k)))


lemma HasUnifEigenvector.hasUnifEigenvalue {f : End R M} {μ : R} {k : ℕ∞} {x : M}
    (h : f.HasUnifEigenvector μ k x) : f.HasUnifEigenvalue μ k := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : ENat
    x : M
    h : f.HasUnifEigenvector μ k x
    ⊢ f.HasUnifEigenvalue μ k
  -/
  rw [HasUnifEigenvalue, Submodule.ne_bot_iff]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : ENat
    x : M
    h : f.HasUnifEigenvector μ k x
    ⊢ Exists fun x => And (Membership.mem ((f.genEigenspace μ) k) x) (Ne x 0)
  -/
  use x; exact h
         /-
           🎉 no goals
         -/


lemma HasUnifEigenvector.apply_eq_smul {f : End R M} {μ : R} {x : M}
    (hx : f.HasUnifEigenvector μ 1 x) : f x = μ • x :=
  mem_genEigenspace_one.mp hx.1


lemma HasUnifEigenvector.pow_apply {f : End R M} {μ : R} {v : M} (hv : f.HasUnifEigenvector μ 1 v)
    (n : ℕ) : (f ^ n) v = μ ^ n • v := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    v : M
    hv : f.HasUnifEigenvector μ 1 v
    n : Nat
    ⊢ Eq ((HPow.hPow f n) v) (HSMul.hSMul (HPow.hPow μ n) v)
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, pow_succ f, hv.apply_eq_smul, smul_smul, pow_succ' μ]
                  /-
                    🎉 no goals
                  -/


theorem HasUnifEigenvalue.exists_hasUnifEigenvector
    {f : End R M} {μ : R} {k : ℕ∞} (hμ : f.HasUnifEigenvalue μ k) :
    ∃ v, f.HasUnifEigenvector μ k v :=
  Submodule.exists_mem_ne_zero_of_ne_bot hμ


lemma HasUnifEigenvalue.pow {f : End R M} {μ : R} (h : f.HasUnifEigenvalue μ 1) (n : ℕ) :
    (f ^ n).HasUnifEigenvalue (μ ^ n) 1 := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    h : f.HasUnifEigenvalue μ 1
    n : Nat
    ⊢ (HPow.hPow f n).HasUnifEigenvalue (HPow.hPow μ n) 1
  -/
  rw [HasUnifEigenvalue, Submodule.ne_bot_iff]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    h : f.HasUnifEigenvalue μ 1
    n : Nat
    ⊢ Exists fun x => And (Membership.mem (((HPow.hPow f n).genEigenspace (HPow.hP …
  -/
  obtain ⟨m : M, hm⟩ := h.exists_hasUnifEigenvector
  /-
    case intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    h : f.HasUnifEigenvalue μ 1
    n : Nat
    m : M
    hm : f.HasUnifEigenvector μ 1 m
    ⊢ Exists fun x => And (Membership.mem (((HPow.hPow f n).genEigenspace (HPow.hP …
  -/
  exact ⟨m, by simpa [mem_genEigenspace_one] using hm.pow_apply n, hm.2⟩
  /-
    🎉 no goals
  -/


/-- A nilpotent endomorphism has nilpotent eigenvalues.

See also `LinearMap.isNilpotent_trace_of_isNilpotent`. -/
lemma HasUnifEigenvalue.isNilpotent_of_isNilpotent [NoZeroSMulDivisors R M] {f : End R M}
    (hfn : IsNilpotent f) {μ : R} (hf : f.HasUnifEigenvalue μ 1) :
    IsNilpotent μ := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    hfn : IsNilpotent f
    μ : R
    hf : f.HasUnifEigenvalue μ 1
    ⊢ IsNilpotent μ
  -/
  obtain ⟨m : M, hm⟩ := hf.exists_hasUnifEigenvector
  /-
    case intro
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    hfn : IsNilpotent f
    μ : R
    hf : f.HasUnifEigenvalue μ 1
    m : M
    hm : f.HasUnifEigenvector μ 1 m
    ⊢ IsNilpotent μ
  -/
  obtain ⟨n : ℕ, hn : f ^ n = 0⟩ := hfn
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ : R
    hf : f.HasUnifEigenvalue μ 1
    m : M
    hm : f.HasUnifEigenvector μ 1 m
    n : Nat
    hn : Eq (HPow.hPow f n) 0
    ⊢ IsNilpotent μ
  -/
  exact ⟨n, by simpa [hn, hm.2, eq_comm (a := (0 : M))] using hm.pow_apply n⟩
  /-
    🎉 no goals
  -/


lemma HasUnifEigenvalue.mem_spectrum {f : End R M} {μ : R} (hμ : HasUnifEigenvalue f μ 1) :
    μ ∈ spectrum R f := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    hμ : f.HasUnifEigenvalue μ 1
    ⊢ Membership.mem (spectrum R f) μ
  -/
  refine spectrum.mem_iff.mpr fun h_unit ↦ ?_
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    hμ : f.HasUnifEigenvalue μ 1
    h_unit : IsUnit (HSub.hSub ((algebraMap R (Module.End R M)) μ) f)
    ⊢ False
  -/
  set f' := LinearMap.GeneralLinearGroup.toLinearEquiv h_unit.unit
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    hμ : f.HasUnifEigenvalue μ 1
    h_unit : IsUnit (HSub.hSub ((algebraMap R (Module.End R M)) μ) f)
    f' : LinearEquiv (RingHom.id R) M M := LinearMap.GeneralLinearGroup.toLinearEq …
    ⊢ False
  -/
  rcases hμ.exists_hasUnifEigenvector with ⟨v, hv⟩
  /-
    case intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    hμ : f.HasUnifEigenvalue μ 1
    h_unit : IsUnit (HSub.hSub ((algebraMap R (Module.End R M)) μ) f)
    f' : LinearEquiv (RingHom.id R) M M := LinearMap.GeneralLinearGroup.toLinearEq …
    v : M
    hv : f.HasUnifEigenvector μ 1 v
    ⊢ False
  -/
  refine hv.2 ((LinearMap.ker_eq_bot'.mp f'.ker) v (?_ : μ • v - f v = 0))
  /-
    case intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    hμ : f.HasUnifEigenvalue μ 1
    h_unit : IsUnit (HSub.hSub ((algebraMap R (Module.End R M)) μ) f)
    f' : LinearEquiv (RingHom.id R) M M := LinearMap.GeneralLinearGroup.toLinearEq …
    v : M
    hv : f.HasUnifEigenvector μ 1 v
    ⊢ Eq (HSub.hSub (HSMul.hSMul μ v) (f v)) 0
  -/
  rw [hv.apply_eq_smul, sub_self]
  /-
    🎉 no goals
  -/


lemma hasUnifEigenvalue_iff_mem_spectrum [FiniteDimensional K V] {f : End K V} {μ : K} :
    f.HasUnifEigenvalue μ 1 ↔ μ ∈ spectrum K f := by
  rw [spectrum.mem_iff, IsUnit.sub_iff, LinearMap.isUnit_iff_ker_eq_bot,
    HasUnifEigenvalue, genEigenspace_one, ne_eq, not_iff_not]
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    ⊢ Iff (Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot) (Eq (Linear …
  -/
  simp [Submodule.ext_iff, LinearMap.mem_ker]
  /-
    🎉 no goals
  -/


alias ⟨_, HasUnifEigenvalue.of_mem_spectrum⟩ := hasUnifEigenvalue_iff_mem_spectrum


lemma genEigenspace_div (f : End K V) (a b : K) (hb : b ≠ 0) :
    genEigenspace f (a / b) 1 = LinearMap.ker (b • f - a • 1) :=
  calc
                                                                  /-
                                                                    K : Type v
                                                                    V : Type w
                                                                    inst✝² : Field K
                                                                    inst✝¹ : AddCommGroup V
                                                                    inst✝ : Module K V
                                                                    f : Module.End K V
                                                                    a b : K
                                                                    hb : Ne b 0
                                                                    ⊢ Eq ((f.genEigenspace (HDiv.hDiv a b)) 1) ((f.genEigenspace (HMul.hMul (Inv.i …
                                                                  -/
    genEigenspace f (a / b) 1 = genEigenspace f (b⁻¹ * a) 1 := by rw [div_eq_mul_inv, mul_comm]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                                    /-
                                                      K : Type v
                                                      V : Type w
                                                      inst✝² : Field K
                                                      inst✝¹ : AddCommGroup V
                                                      inst✝ : Module K V
                                                      f : Module.End K V
                                                      a b : K
                                                      hb : Ne b 0
                                                      ⊢ Eq ((f.genEigenspace (HMul.hMul (Inv.inv b) a)) 1) (LinearMap.ker (HSub.hSub …
                                                    -/
    _ = LinearMap.ker (f - (b⁻¹ * a) • 1)     := by rw [genEigenspace_one]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      K : Type v
                                                      V : Type w
                                                      inst✝² : Field K
                                                      inst✝¹ : AddCommGroup V
                                                      inst✝ : Module K V
                                                      f : Module.End K V
                                                      a b : K
                                                      hb : Ne b 0
                                                      ⊢ Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul (HMul.hMul (Inv.inv b) a) 1))) ( …
                                                    -/
    _ = LinearMap.ker (f - b⁻¹ • a • 1)       := by rw [smul_smul]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      K : Type v
                                                      V : Type w
                                                      inst✝² : Field K
                                                      inst✝¹ : AddCommGroup V
                                                      inst✝ : Module K V
                                                      f : Module.End K V
                                                      a b : K
                                                      hb : Ne b 0
                                                      ⊢ Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul (Inv.inv b) (HSMul.hSMul a 1)))) …
                                                    -/
    _ = LinearMap.ker (b • (f - b⁻¹ • a • 1)) := by rw [LinearMap.ker_smul _ b hb]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      K : Type v
                                                      V : Type w
                                                      inst✝² : Field K
                                                      inst✝¹ : AddCommGroup V
                                                      inst✝ : Module K V
                                                      f : Module.End K V
                                                      a b : K
                                                      hb : Ne b 0
                                                      ⊢ Eq (LinearMap.ker (HSMul.hSMul b (HSub.hSub f (HSMul.hSMul (Inv.inv b) (HSMu …
                                                    -/
    _ = LinearMap.ker (b • f - a • 1)         := by rw [smul_sub, smul_inv_smul₀ hb]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The generalized eigenrange for a linear map `f`, a scalar `μ`, and an exponent `k ∈ ℕ∞`
is the range of `(f - μ • id) ^ k` if `k` is a natural number,
or the infimum of these ranges if `k = ∞`. -/
def genEigenrange (f : End R M) (μ : R) (k : ℕ∞) : Submodule R M :=
  ⨅ l : ℕ, ⨅ (_ : l ≤ k), LinearMap.range ((f - μ • 1) ^ l)


lemma genEigenrange_nat {f : End R M} {μ : R} {k : ℕ} :
    f.genEigenrange μ k = LinearMap.range ((f - μ • 1) ^ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    ⊢ Eq (f.genEigenrange μ ↑k) (LinearMap.range (HPow.hPow (HSub.hSub f (HSMul.hS …
  -/
  ext x
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    x : M
    ⊢ Iff (Membership.mem (f.genEigenrange μ ↑k) x) (Membership.mem (LinearMap.ran …
  -/
  simp only [genEigenrange, Nat.cast_le, Submodule.mem_iInf, LinearMap.mem_range]
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    x : M
    ⊢ Iff (∀ (i : Nat), LE.le i k → Exists fun y => Eq ((HPow.hPow (HSub.hSub f (H …
  -/
  constructor
    /-
      case h.mp
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      ⊢ (∀ (i : Nat), LE.le i k → Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMul …
    -/
  · intro h
    /-
      case h.mp
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      h : ∀ (i : Nat), LE.le i k → Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMu …
      ⊢ Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k) y) x
    -/
    exact h _ le_rfl
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      ⊢ (Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k) y) x) → ∀ …
    -/
  · rintro ⟨x, rfl⟩ i hi
    /-
      case h.mpr.intro
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      i : Nat
      hi : LE.le i k
      ⊢ Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) i) y) ((HPow. …
    -/
    have : k = i + (k - i) := by omega
    /-
      case h.mpr.intro
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      i : Nat
      hi : LE.le i k
      this : Eq k (HAdd.hAdd i (HSub.hSub k i))
      ⊢ Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) i) y) ((HPow. …
    -/
    rw [this, pow_add]
    /-
      case h.mpr.intro
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      μ : R
      k : Nat
      x : M
      i : Nat
      hi : LE.le i k
      this : Eq k (HAdd.hAdd i (HSub.hSub k i))
      ⊢ Exists fun y => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) i) y) ((HMul. …
    -/
    exact ⟨_, rfl⟩
    /-
      🎉 no goals
    -/


/-- The exponent of a generalized eigenvalue is never 0. -/
lemma HasUnifEigenvalue.exp_ne_zero {f : End R M} {μ : R} {k : ℕ}
    (h : f.HasUnifEigenvalue μ k) : k ≠ 0 := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    h : f.HasUnifEigenvalue μ ↑k
    ⊢ Ne k 0
  -/
  rintro rfl
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    h : f.HasUnifEigenvalue μ ↑0
    ⊢ False
  -/
  simp [HasUnifEigenvalue, Nat.cast_zero, genEigenspace_zero] at h
  /-
    🎉 no goals
  -/


/-- If there exists a natural number `k` such that the kernel of `(f - μ • id) ^ k` is the
maximal generalized eigenspace, then this value is the least such `k`. If not, this value is not
meaningful. -/
noncomputable def maxUnifEigenspaceIndex (f : End R M) (μ : R) :=
  monotonicSequenceLimitIndex <| (f.genEigenspace μ).comp <| WithTop.coeOrderHom.toOrderHom


/-- For an endomorphism of a Noetherian module, the maximal eigenspace is always of the form kernel
`(f - μ • id) ^ k` for some `k`. -/
lemma genEigenspace_top_eq_maxUnifEigenspaceIndex [h : IsNoetherian R M] (f : End R M) (μ : R) :
    genEigenspace f μ ⊤ = f.genEigenspace μ (maxUnifEigenspaceIndex f μ) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : IsNoetherian R M
    f : Module.End R M
    μ : R
    ⊢ Eq ((f.genEigenspace μ) Top.top) ((f.genEigenspace μ) ↑(f.maxUnifEigenspaceI …
  -/
  rw [isNoetherian_iff] at h
  have := WellFounded.iSup_eq_monotonicSequenceLimit h <|
    (f.genEigenspace μ).comp <| WithTop.coeOrderHom.toOrderHom
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    f : Module.End R M
    μ : R
    this : Eq (iSup ⇑((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom)) (m …
    ⊢ Eq ((f.genEigenspace μ) Top.top) ((f.genEigenspace μ) ↑(f.maxUnifEigenspaceI …
  -/
  convert this using 1
  simp only [genEigenspace, OrderHom.coe_mk, le_top, iSup_pos, OrderHom.comp_coe,
    Function.comp_def]
  /-
    case h.e'_2
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    f : Module.End R M
    μ : R
    this : Eq (iSup ⇑((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom)) (m …
    ⊢ Eq (iSup fun l => LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) l …
  -/
  rw [iSup_prod', iSup_subtype', ← sSup_range, ← sSup_range]
  /-
    case h.e'_2
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    f : Module.End R M
    μ : R
    this : Eq (iSup ⇑((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom)) (m …
    ⊢ Eq (SupSet.sSup (Set.range fun l => LinearMap.ker (HPow.hPow (HSub.hSub f (H …
  -/
  congr
  /-
    case h.e'_2.e_a
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    f : Module.End R M
    μ : R
    this : Eq (iSup ⇑((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom)) (m …
    ⊢ Eq (Set.range fun l => LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ  …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma genEigenspace_le_genEigenspace_maxUnifEigenspaceIndex [IsNoetherian R M] (f : End R M)
    (μ : R) (k : ℕ∞) :
    f.genEigenspace μ k ≤ f.genEigenspace μ (maxUnifEigenspaceIndex f μ) := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    k : ENat
    ⊢ LE.le ((f.genEigenspace μ) k) ((f.genEigenspace μ) ↑(f.maxUnifEigenspaceInde …
  -/
  rw [← genEigenspace_top_eq_maxUnifEigenspaceIndex]
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    k : ENat
    ⊢ LE.le ((f.genEigenspace μ) k) ((f.genEigenspace μ) Top.top)
  -/
  exact (f.genEigenspace μ).monotone le_top
  /-
    🎉 no goals
  -/


/-- Generalized eigenspaces for exponents at least `finrank K V` are equal to each other. -/
theorem genEigenspace_eq_genEigenspace_maxUnifEigenspaceIndex_of_le [IsNoetherian R M]
    (f : End R M) (μ : R) {k : ℕ} (hk : maxUnifEigenspaceIndex f μ ≤ k) :
    f.genEigenspace μ k = f.genEigenspace μ (maxUnifEigenspaceIndex f μ) :=
  le_antisymm
    (genEigenspace_le_genEigenspace_maxUnifEigenspaceIndex _ _ _)
                                        /-
                                          R : Type v
                                          M : Type w
                                          inst✝³ : CommRing R
                                          inst✝² : AddCommGroup M
                                          inst✝¹ : Module R M
                                          inst✝ : IsNoetherian R M
                                          f : Module.End R M
                                          μ : R
                                          k : Nat
                                          hk : LE.le (f.maxUnifEigenspaceIndex μ) k
                                          ⊢ LE.le ↑(f.maxUnifEigenspaceIndex μ) ↑k
                                        -/
    ((f.genEigenspace μ).monotone <| by simpa using hk)
                                        /-
                                          🎉 no goals
                                        -/


/-- A generalized eigenvalue for some exponent `k` is also
    a generalized eigenvalue for exponents larger than `k`. -/
lemma HasUnifEigenvalue.le {f : End R M} {μ : R} {k m : ℕ∞}
    (hm : k ≤ m) (hk : f.HasUnifEigenvalue μ k) :
    f.HasUnifEigenvalue μ m := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LE.le k m
    hk : f.HasUnifEigenvalue μ k
    ⊢ f.HasUnifEigenvalue μ m
  -/
  unfold HasUnifEigenvalue at *
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LE.le k m
    hk : Ne ((f.genEigenspace μ) k) Bot.bot
    ⊢ Ne ((f.genEigenspace μ) m) Bot.bot
  -/
  contrapose! hk
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LE.le k m
    hk : Eq ((f.genEigenspace μ) m) Bot.bot
    ⊢ Eq ((f.genEigenspace μ) k) Bot.bot
  -/
  rw [← le_bot_iff, ← hk]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LE.le k m
    hk : Eq ((f.genEigenspace μ) m) Bot.bot
    ⊢ LE.le ((f.genEigenspace μ) k) ((f.genEigenspace μ) m)
  -/
  exact (f.genEigenspace _).monotone hm
  /-
    🎉 no goals
  -/


/-- A generalized eigenvalue for some exponent `k` is also
    a generalized eigenvalue for positive exponents. -/
lemma HasUnifEigenvalue.lt {f : End R M} {μ : R} {k m : ℕ∞}
    (hm : 0 < m) (hk : f.HasUnifEigenvalue μ k) :
    f.HasUnifEigenvalue μ m := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    ⊢ f.HasUnifEigenvalue μ m
  -/
  apply HasUnifEigenvalue.le (k := 1) (Order.one_le_iff_pos.mpr hm)
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    ⊢ f.HasUnifEigenvalue μ 1
  -/
  intro contra; apply hk
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Eq ((f.genEigenspace μ) 1) Bot.bot
    ⊢ Eq ((f.genEigenspace μ) k) Bot.bot
  -/
  rw [genEigenspace_one, LinearMap.ker_eq_bot] at contra
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    ⊢ Eq ((f.genEigenspace μ) k) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    ⊢ LE.le ((f.genEigenspace μ) k) Bot.bot
  -/
  intro x hx
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    x : M
    hx : Membership.mem ((f.genEigenspace μ) k) x
    ⊢ Membership.mem Bot.bot x
  -/
  rw [mem_genEigenspace] at hx
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    x : M
    hx : Exists fun l => And (LE.le (↑l) k) (Membership.mem (LinearMap.ker (HPow.h …
    ⊢ Membership.mem Bot.bot x
  -/
  rcases hx with ⟨l, -, hx⟩
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    x : M
    l : Nat
    hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
    ⊢ Membership.mem Bot.bot x
  -/
  rwa [LinearMap.ker_eq_bot.mpr] at hx
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    x : M
    l : Nat
    hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
    ⊢ Function.Injective ⇑(HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) l)
  -/
  rw [LinearMap.coe_pow (f - μ • 1) l]
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k m : ENat
    hm : LT.lt 0 m
    hk : f.HasUnifEigenvalue μ k
    contra : Function.Injective ⇑(HSub.hSub f (HSMul.hSMul μ 1))
    x : M
    l : Nat
    hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
    ⊢ Function.Injective (Nat.iterate (⇑(HSub.hSub f (HSMul.hSMul μ 1))) l)
  -/
  exact Function.Injective.iterate contra l
  /-
    🎉 no goals
  -/


/-- Generalized eigenvalues are actually just eigenvalues. -/
@[simp]
lemma hasUnifEigenvalue_iff_hasUnifEigenvalue_one {f : End R M} {μ : R} {k : ℕ∞} (hk : 0 < k) :
    f.HasUnifEigenvalue μ k ↔ f.HasUnifEigenvalue μ 1 :=
  ⟨HasUnifEigenvalue.lt zero_lt_one, HasUnifEigenvalue.lt hk⟩


lemma maxUnifEigenspaceIndex_le_finrank [FiniteDimensional K V] (f : End K V) (μ : K) :
    maxUnifEigenspaceIndex f μ ≤ finrank K V := by
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    ⊢ LE.le (f.maxUnifEigenspaceIndex μ) (Module.finrank K V)
  -/
  apply Nat.sInf_le
  /-
    case hm
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    ⊢ Membership.mem (setOf fun n => ∀ (m : Nat), LE.le n m → Eq (((f.genEigenspac …
  -/
  intro n hn
  /-
    case hm
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    n : Nat
    hn : LE.le (Module.finrank K V) n
    ⊢ Eq (((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom) (Module.finran …
  -/
  apply le_antisymm
    /-
      case hm.a
      K : Type v
      V : Type w
      inst✝³ : Field K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      μ : K
      n : Nat
      hn : LE.le (Module.finrank K V) n
      ⊢ LE.le (((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom) (Module.fin …
    -/
  · exact (f.genEigenspace μ).monotone <| WithTop.coeOrderHom.monotone hn
    /-
      🎉 no goals
    -/
    /-
      case hm.a
      K : Type v
      V : Type w
      inst✝³ : Field K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      μ : K
      n : Nat
      hn : LE.le (Module.finrank K V) n
      ⊢ LE.le (((f.genEigenspace μ).comp WithTop.coeOrderHom.toOrderHom) n) (((f.gen …
    -/
  · show (f.genEigenspace μ) n ≤ (f.genEigenspace μ) (finrank K V)
    /-
      case hm.a
      K : Type v
      V : Type w
      inst✝³ : Field K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      μ : K
      n : Nat
      hn : LE.le (Module.finrank K V) n
      ⊢ LE.le ((f.genEigenspace μ) ↑n) ((f.genEigenspace μ) ↑(Module.finrank K V))
    -/
    rw [genEigenspace_nat, genEigenspace_nat]
    /-
      case hm.a
      K : Type v
      V : Type w
      inst✝³ : Field K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      μ : K
      n : Nat
      hn : LE.le (Module.finrank K V) n
      ⊢ LE.le (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) n)) (LinearM …
    -/
    apply ker_pow_le_ker_pow_finrank
    /-
      🎉 no goals
    -/


/-- Every generalized eigenvector is a generalized eigenvector for exponent `finrank K V`.
    (Lemma 8.11 of [axler2015]) -/
lemma genEigenspace_le_genEigenspace_finrank [FiniteDimensional K V] (f : End K V)
    (μ : K) (k : ℕ∞) : f.genEigenspace μ k ≤ f.genEigenspace μ (finrank K V) := by
  calc f.genEigenspace μ k
      ≤ f.genEigenspace μ ⊤ := (f.genEigenspace _).monotone le_top
    _ ≤ f.genEigenspace μ (finrank K V) := by
      rw [genEigenspace_top_eq_maxUnifEigenspaceIndex]
      exact (f.genEigenspace _).monotone <| by simpa using maxUnifEigenspaceIndex_le_finrank f μ


/-- Generalized eigenspaces for exponents at least `finrank K V` are equal to each other. -/
theorem genEigenspace_eq_genEigenspace_finrank_of_le [FiniteDimensional K V]
    (f : End K V) (μ : K) {k : ℕ} (hk : finrank K V ≤ k) :
    f.genEigenspace μ k = f.genEigenspace μ (finrank K V) :=
  le_antisymm
    (genEigenspace_le_genEigenspace_finrank _ _ _)
                                        /-
                                          K : Type v
                                          V : Type w
                                          inst✝³ : Field K
                                          inst✝² : AddCommGroup V
                                          inst✝¹ : Module K V
                                          inst✝ : FiniteDimensional K V
                                          f : Module.End K V
                                          μ : K
                                          k : Nat
                                          hk : LE.le (Module.finrank K V) k
                                          ⊢ LE.le ↑(Module.finrank K V) ↑k
                                        -/
    ((f.genEigenspace μ).monotone <| by simpa using hk)
                                        /-
                                          🎉 no goals
                                        -/


lemma mapsTo_genEigenspace_of_comm {f g : End R M} (h : Commute f g) (μ : R) (k : ℕ∞) :
    MapsTo g (f.genEigenspace μ k) (f.genEigenspace μ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    h : Commute f g
    μ : R
    k : ENat
    ⊢ Set.MapsTo ⇑g ↑((f.genEigenspace μ) k) ↑((f.genEigenspace μ) k)
  -/
  intro x hx
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    h : Commute f g
    μ : R
    k : ENat
    x : M
    hx : Membership.mem (↑((f.genEigenspace μ) k)) x
    ⊢ Membership.mem (↑((f.genEigenspace μ) k)) (g x)
  -/
  simp only [SetLike.mem_coe, mem_genEigenspace, LinearMap.mem_ker] at hx ⊢
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    h : Commute f g
    μ : R
    k : ENat
    x : M
    hx : Exists fun l => And (LE.le (↑l) k) (Eq ((HPow.hPow (HSub.hSub f (HSMul.hS …
    ⊢ Exists fun l => And (LE.le (↑l) k) (Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul …
  -/
  rcases hx with ⟨l, hl, hx⟩
  replace h : Commute ((f - μ • (1 : End R M)) ^ l) g :=
    (h.sub_left <| Algebra.commute_algebraMap_left μ g).pow_left l
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    μ : R
    k : ENat
    x : M
    l : Nat
    hl : LE.le (↑l) k
    hx : Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) l) x) 0
    h : Commute (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) l) g
    ⊢ Exists fun l => And (LE.le (↑l) k) (Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul …
  -/
  use l, hl
  rw [← LinearMap.comp_apply, ← LinearMap.mul_eq_comp, h.eq, LinearMap.mul_eq_comp,
    LinearMap.comp_apply, hx, map_zero]


/-- The restriction of `f - μ • 1` to the `k`-fold generalized `μ`-eigenspace is nilpotent. -/
lemma isNilpotent_restrict_genEigenspace_nat (f : End R M) (μ : R) (k : ℕ)
    (h : MapsTo (f - μ • (1 : End R M))
      (f.genEigenspace μ k) (f.genEigenspace μ k) :=
      mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ) μ k) :
    IsNilpotent ((f - μ • 1).restrict h) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    ⊢ IsNilpotent (LinearMap.restrict (HSub.hSub f (HSMul.hSMul μ 1)) h)
  -/
  use k
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    ⊢ Eq (HPow.hPow (LinearMap.restrict (HSub.hSub f (HSMul.hSMul μ 1)) h) k) 0
  -/
  ext ⟨x, hx⟩
  /-
    case h.h.mk.a
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    x : M
    hx : Membership.mem ((f.genEigenspace μ) ↑k) x
    ⊢ Eq ↑((HPow.hPow (LinearMap.restrict (HSub.hSub f (HSMul.hSMul μ 1)) h) k) ⟨x …
  -/
  rw [mem_genEigenspace_nat] at hx
  rw [LinearMap.zero_apply, ZeroMemClass.coe_zero, ZeroMemClass.coe_eq_zero,
    LinearMap.pow_restrict, LinearMap.restrict_apply]
  /-
    case h.h.mk.a
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    x : M
    hx✝ : Membership.mem ((f.genEigenspace μ) ↑k) x
    hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
    ⊢ Eq ⟨(HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k) ↑⟨x, hx✝⟩, ⋯⟩ 0
  -/
  ext
  /-
    case h.h.mk.a.a
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    k : Nat
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    x : M
    hx✝ : Membership.mem ((f.genEigenspace μ) ↑k) x
    hx : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
    ⊢ Eq ↑⟨(HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) k) ↑⟨x, hx✝⟩, ⋯⟩ ↑0
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- The restriction of `f - μ • 1` to the generalized `μ`-eigenspace is nilpotent. -/
lemma isNilpotent_restrict_genEigenspace_top [IsNoetherian R M] (f : End R M) (μ : R)
    (h : MapsTo (f - μ • (1 : End R M))
      (f.genEigenspace μ ⊤) (f.genEigenspace μ ⊤) :=
      mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ) μ _) :
    IsNilpotent ((f - μ • 1).restrict h) := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    ⊢ IsNilpotent (LinearMap.restrict (HSub.hSub f (HSMul.hSMul μ 1)) h)
  -/
  apply isNilpotent_restrict_of_le
  /-
    case h
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    ⊢ LE.le ((f.genEigenspace μ) Top.top) ?q
  -/
  on_goal 2 => apply isNilpotent_restrict_genEigenspace_nat f μ (maxUnifEigenspaceIndex f μ)
  /-
    case h
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f (HSMul.hSMul μ 1)) ↑((f.genEigenspace μ …
    ⊢ LE.le ((f.genEigenspace μ) Top.top) ((f.genEigenspace μ) ↑(f.maxUnifEigenspa …
  -/
  rw [genEigenspace_top_eq_maxUnifEigenspaceIndex]
  /-
    🎉 no goals
  -/


/-- The submodule `eigenspace f μ` for a linear map `f` and a scalar `μ` consists of all vectors `x`
    such that `f x = μ • x`. (Def 5.36 of [axler2015])-/
abbrev eigenspace (f : End R M) (μ : R) : Submodule R M :=
  f.genEigenspace μ 1


lemma eigenspace_def {f : End R M} {μ : R} :
    f.eigenspace μ = LinearMap.ker (f - μ • 1) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ Eq (f.eigenspace μ) (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1)))
  -/
  rw [eigenspace, genEigenspace_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem eigenspace_zero (f : End R M) : f.eigenspace 0 = LinearMap.ker f := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    ⊢ Eq (f.eigenspace 0) (LinearMap.ker f)
  -/
  simp only [eigenspace, ← Nat.cast_one (R := ℕ∞), genEigenspace_zero_nat, pow_one]
  /-
    🎉 no goals
  -/


/-- A nonzero element of an eigenspace is an eigenvector. (Def 5.7 of [axler2015]) -/
abbrev HasEigenvector (f : End R M) (μ : R) (x : M) : Prop :=
  HasUnifEigenvector f μ 1 x


lemma hasEigenvector_iff {f : End R M} {μ : R} {x : M} :
    f.HasEigenvector μ x ↔ x ∈ f.eigenspace μ ∧ x ≠ 0 := Iff.rfl


/-- A scalar `μ` is an eigenvalue for a linear map `f` if there are nonzero vectors `x`
    such that `f x = μ • x`. (Def 5.5 of [axler2015]) -/
abbrev HasEigenvalue (f : End R M) (a : R) : Prop :=
  HasUnifEigenvalue f a 1


lemma hasEigenvalue_iff {f : End R M} {μ : R} :
    f.HasEigenvalue μ ↔ f.eigenspace μ ≠ ⊥ := Iff.rfl


/-- The eigenvalues of the endomorphism `f`, as a subtype of `R`. -/
abbrev Eigenvalues (f : End R M) : Type _ :=
  UnifEigenvalues f 1


@[coe]
abbrev Eigenvalues.val (f : Module.End R M) : Eigenvalues f → R := UnifEigenvalues.val f 1


theorem hasEigenvalue_of_hasEigenvector {f : End R M} {μ : R} {x : M} (h : HasEigenvector f μ x) :
    HasEigenvalue f μ :=
  h.hasUnifEigenvalue


theorem mem_eigenspace_iff {f : End R M} {μ : R} {x : M} : x ∈ eigenspace f μ ↔ f x = μ • x :=
  mem_genEigenspace_one


nonrec
theorem HasEigenvector.apply_eq_smul {f : End R M} {μ : R} {x : M} (hx : f.HasEigenvector μ x) :
    f x = μ • x :=
  hx.apply_eq_smul


nonrec
theorem HasEigenvector.pow_apply {f : End R M} {μ : R} {v : M} (hv : f.HasEigenvector μ v) (n : ℕ) :
    (f ^ n) v = μ ^ n • v :=
  hv.pow_apply n


theorem HasEigenvalue.exists_hasEigenvector {f : End R M} {μ : R} (hμ : f.HasEigenvalue μ) :
    ∃ v, f.HasEigenvector μ v :=
  Submodule.exists_mem_ne_zero_of_ne_bot hμ


nonrec
lemma HasEigenvalue.pow {f : End R M} {μ : R} (h : f.HasEigenvalue μ) (n : ℕ) :
    (f ^ n).HasEigenvalue (μ ^ n) :=
  h.pow n


/-- A nilpotent endomorphism has nilpotent eigenvalues.

See also `LinearMap.isNilpotent_trace_of_isNilpotent`. -/
nonrec
lemma HasEigenvalue.isNilpotent_of_isNilpotent [NoZeroSMulDivisors R M] {f : End R M}
    (hfn : IsNilpotent f) {μ : R} (hf : f.HasEigenvalue μ) :
    IsNilpotent μ :=
  hf.isNilpotent_of_isNilpotent hfn


nonrec
theorem HasEigenvalue.mem_spectrum {f : End R M} {μ : R} (hμ : HasEigenvalue f μ) :
    μ ∈ spectrum R f :=
  hμ.mem_spectrum


theorem hasEigenvalue_iff_mem_spectrum [FiniteDimensional K V] {f : End K V} {μ : K} :
    f.HasEigenvalue μ ↔ μ ∈ spectrum K f :=
  hasUnifEigenvalue_iff_mem_spectrum


alias ⟨_, HasEigenvalue.of_mem_spectrum⟩ := hasEigenvalue_iff_mem_spectrum


theorem eigenspace_div (f : End K V) (a b : K) (hb : b ≠ 0) :
    eigenspace f (a / b) = LinearMap.ker (b • f - algebraMap K (End K V) a) :=
  genEigenspace_div f a b hb


@[deprecated genEigenspace_nat (since := "2024-10-28")]
lemma genEigenspace_def (f : End R M) (μ : R) (k : ℕ) :
    f.genEigenspace μ k = LinearMap.ker ((f - μ • 1) ^ k) :=
  genEigenspace_nat


/-- A nonzero element of a generalized eigenspace is a generalized eigenvector.
    (Def 8.9 of [axler2015])-/
abbrev HasGenEigenvector (f : End R M) (μ : R) (k : ℕ) (x : M) : Prop :=
  HasUnifEigenvector f μ k x


lemma hasGenEigenvector_iff {f : End R M} {μ : R} {k : ℕ} {x : M} :
    f.HasGenEigenvector μ k x ↔ x ∈ f.genEigenspace μ k ∧ x ≠ 0 := Iff.rfl


/-- A scalar `μ` is a generalized eigenvalue for a linear map `f` and an exponent `k ∈ ℕ` if there
    are generalized eigenvectors for `f`, `k`, and `μ`. -/
abbrev HasGenEigenvalue (f : End R M) (μ : R) (k : ℕ) : Prop :=
  HasUnifEigenvalue f μ k


lemma hasGenEigenvalue_iff {f : End R M} {μ : R} {k : ℕ} :
    f.HasGenEigenvalue μ k ↔ f.genEigenspace μ k ≠ ⊥ := Iff.rfl


@[deprecated genEigenrange_nat (since := "2024-10-28")]
lemma genEigenrange_def {f : End R M} {μ : R} {k : ℕ} :
    f.genEigenrange μ k = LinearMap.range ((f - μ • 1) ^ k) :=
  genEigenrange_nat


/-- The exponent of a generalized eigenvalue is never 0. -/
theorem exp_ne_zero_of_hasGenEigenvalue {f : End R M} {μ : R} {k : ℕ}
    (h : f.HasGenEigenvalue μ k) : k ≠ 0 :=
  HasUnifEigenvalue.exp_ne_zero h


/-- The union of the kernels of `(f - μ • id) ^ k` over all `k`. -/
abbrev maxGenEigenspace (f : End R M) (μ : R) : Submodule R M :=
  genEigenspace f μ ⊤


lemma iSup_genEigenspace_eq (f : End R M) (μ : R) :
    ⨆ k : ℕ, (f.genEigenspace μ) k = f.maxGenEigenspace μ := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ : R
    ⊢ Eq (iSup fun k => (f.genEigenspace μ) ↑k) (f.maxGenEigenspace μ)
  -/
  simp_rw [maxGenEigenspace, genEigenspace_top]
  /-
    🎉 no goals
  -/


@[deprecated iSup_genEigenspace_eq (since := "2024-10-23")]
lemma maxGenEigenspace_def (f : End R M) (μ : R) :
    f.maxGenEigenspace μ = ⨆ k : ℕ, f.genEigenspace μ k :=
  (iSup_genEigenspace_eq f μ).symm


theorem genEigenspace_le_maximal (f : End R M) (μ : R) (k : ℕ) :
    f.genEigenspace μ k ≤ f.maxGenEigenspace μ :=
  (f.genEigenspace μ).monotone le_top


@[simp]
theorem mem_maxGenEigenspace (f : End R M) (μ : R) (m : M) :
    m ∈ f.maxGenEigenspace μ ↔ ∃ k : ℕ, ((f - μ • (1 : End R M)) ^ k) m = 0 :=
  mem_genEigenspace_top


/-- If there exists a natural number `k` such that the kernel of `(f - μ • id) ^ k` is the
maximal generalized eigenspace, then this value is the least such `k`. If not, this value is not
meaningful. -/
noncomputable abbrev maxGenEigenspaceIndex (f : End R M) (μ : R) :=
  maxUnifEigenspaceIndex f μ


/-- For an endomorphism of a Noetherian module, the maximal eigenspace is always of the form kernel
`(f - μ • id) ^ k` for some `k`. -/
theorem maxGenEigenspace_eq [IsNoetherian R M] (f : End R M) (μ : R) :
    maxGenEigenspace f μ = f.genEigenspace μ (maxGenEigenspaceIndex f μ) :=
  genEigenspace_top_eq_maxUnifEigenspaceIndex _ _


/-- A generalized eigenvalue for some exponent `k` is also
    a generalized eigenvalue for exponents larger than `k`. -/
theorem hasGenEigenvalue_of_hasGenEigenvalue_of_le {f : End R M} {μ : R} {k : ℕ}
    {m : ℕ} (hm : k ≤ m) (hk : f.HasGenEigenvalue μ k) :
    f.HasGenEigenvalue μ m :=
              /-
                R : Type v
                M : Type w
                inst✝² : CommRing R
                inst✝¹ : AddCommGroup M
                inst✝ : Module R M
                f : Module.End R M
                μ : R
                k m : Nat
                hm : LE.le k m
                hk : f.HasGenEigenvalue μ k
                ⊢ LE.le ↑k ↑m
              -/
  hk.le <| by simpa using hm
              /-
                🎉 no goals
              -/


/-- The eigenspace is a subspace of the generalized eigenspace. -/
theorem eigenspace_le_genEigenspace {f : End R M} {μ : R} {k : ℕ} (hk : 0 < k) :
    f.eigenspace μ ≤ f.genEigenspace μ k :=
                                     /-
                                       R : Type v
                                       M : Type w
                                       inst✝² : CommRing R
                                       inst✝¹ : AddCommGroup M
                                       inst✝ : Module R M
                                       f : Module.End R M
                                       μ : R
                                       k : Nat
                                       hk : LT.lt 0 k
                                       ⊢ LE.le 1 ↑k
                                     -/
  (f.genEigenspace _).monotone <| by simpa using Nat.succ_le_of_lt hk
                                     /-
                                       🎉 no goals
                                     -/


/-- All eigenvalues are generalized eigenvalues. -/
theorem hasGenEigenvalue_of_hasEigenvalue {f : End R M} {μ : R} {k : ℕ} (hk : 0 < k)
    (hμ : f.HasEigenvalue μ) : f.HasGenEigenvalue μ k :=
              /-
                R : Type v
                M : Type w
                inst✝² : CommRing R
                inst✝¹ : AddCommGroup M
                inst✝ : Module R M
                f : Module.End R M
                μ : R
                k : Nat
                hk : LT.lt 0 k
                hμ : f.HasEigenvalue μ
                ⊢ LT.lt 0 ↑k
              -/
  hμ.lt <| by simpa using hk
              /-
                🎉 no goals
              -/


/-- All generalized eigenvalues are eigenvalues. -/
theorem hasEigenvalue_of_hasGenEigenvalue {f : End R M} {μ : R} {k : ℕ}
    (hμ : f.HasGenEigenvalue μ k) : f.HasEigenvalue μ :=
  hμ.lt zero_lt_one


/-- Generalized eigenvalues are actually just eigenvalues. -/
@[simp]
theorem hasGenEigenvalue_iff_hasEigenvalue {f : End R M} {μ : R} {k : ℕ} (hk : 0 < k) :
    f.HasGenEigenvalue μ k ↔ f.HasEigenvalue μ :=
                                                    /-
                                                      R : Type v
                                                      M : Type w
                                                      inst✝² : CommRing R
                                                      inst✝¹ : AddCommGroup M
                                                      inst✝ : Module R M
                                                      f : Module.End R M
                                                      μ : R
                                                      k : Nat
                                                      hk : LT.lt 0 k
                                                      ⊢ LT.lt 0 ↑k
                                                    -/
  hasUnifEigenvalue_iff_hasUnifEigenvalue_one <| by simpa using hk
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem maxGenEigenspace_eq_genEigenspace_finrank
    [FiniteDimensional K V] (f : End K V) (μ : K) :
    f.maxGenEigenspace μ = f.genEigenspace μ (finrank K V) := by
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    ⊢ Eq (f.maxGenEigenspace μ) ((f.genEigenspace μ) ↑(Module.finrank K V))
  -/
  apply le_antisymm _ <| (f.genEigenspace μ).monotone le_top
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    ⊢ LE.le ((f.genEigenspace μ) Top.top) ((f.genEigenspace μ) ↑(Module.finrank K  …
  -/
  rw [genEigenspace_top_eq_maxUnifEigenspaceIndex]
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    ⊢ LE.le ((f.genEigenspace μ) ↑(f.maxUnifEigenspaceIndex μ)) ((f.genEigenspace  …
  -/
  apply genEigenspace_le_genEigenspace_finrank f μ
  /-
    🎉 no goals
  -/


lemma mapsTo_maxGenEigenspace_of_comm {f g : End R M} (h : Commute f g) (μ : R) :
    MapsTo g ↑(f.maxGenEigenspace μ) ↑(f.maxGenEigenspace μ) :=
  mapsTo_genEigenspace_of_comm h μ ⊤


@[deprecated mapsTo_iSup_genEigenspace_of_comm (since := "2024-10-23")]
lemma mapsTo_iSup_genEigenspace_of_comm {f g : End R M} (h : Commute f g) (μ : R) :
    MapsTo g ↑(⨆ k : ℕ, f.genEigenspace μ k) ↑(⨆ k : ℕ, f.genEigenspace μ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    h : Commute f g
    μ : R
    ⊢ Set.MapsTo ⇑g ↑(iSup fun k => (f.genEigenspace μ) ↑k) ↑(iSup fun k => (f.gen …
  -/
  rw [iSup_genEigenspace_eq]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    h : Commute f g
    μ : R
    ⊢ Set.MapsTo ⇑g ↑(f.maxGenEigenspace μ) ↑(f.maxGenEigenspace μ)
  -/
  apply mapsTo_maxGenEigenspace_of_comm h
  /-
    🎉 no goals
  -/


/-- The restriction of `f - μ • 1` to the `k`-fold generalized `μ`-eigenspace is nilpotent. -/
lemma isNilpotent_restrict_sub_algebraMap (f : End R M) (μ : R) (k : ℕ)
    (h : MapsTo (f - algebraMap R (End R M) μ)
      (f.genEigenspace μ k) (f.genEigenspace μ k) :=
      mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ) μ k) :
    IsNilpotent ((f - algebraMap R (End R M) μ).restrict h) :=
  isNilpotent_restrict_genEigenspace_nat _ _ _


/-- The restriction of `f - μ • 1` to the generalized `μ`-eigenspace is nilpotent. -/
lemma isNilpotent_restrict_maxGenEigenspace_sub_algebraMap [IsNoetherian R M] (f : End R M) (μ : R)
    (h : MapsTo (f - algebraMap R (End R M) μ)
      ↑(f.maxGenEigenspace μ) ↑(f.maxGenEigenspace μ) :=
      mapsTo_maxGenEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ) μ) :
    IsNilpotent ((f - algebraMap R (End R M) μ).restrict h) := by
  apply isNilpotent_restrict_of_le (q := f.genEigenspace μ (maxUnifEigenspaceIndex f μ))
    _ (isNilpotent_restrict_genEigenspace_nat f μ (maxUnifEigenspaceIndex f μ))
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f ((algebraMap R (Module.End R M)) μ)) ↑( …
    ⊢ LE.le (f.maxGenEigenspace μ) ((f.genEigenspace μ) ↑(f.maxUnifEigenspaceIndex …
  -/
  rw [maxGenEigenspace_eq]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- The restriction of `f - μ • 1` to the generalized `μ`-eigenspace is nilpotent. -/
@[deprecated isNilpotent_restrict_maxGenEigenspace_sub_algebraMap (since := "2024-10-23")]
lemma isNilpotent_restrict_iSup_sub_algebraMap [IsNoetherian R M] (f : End R M) (μ : R)
    (h : MapsTo (f - algebraMap R (End R M) μ)
      ↑(⨆ k : ℕ, f.genEigenspace μ k) ↑(⨆ k : ℕ, f.genEigenspace μ k) :=
      mapsTo_iSup_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ) μ) :
    IsNilpotent ((f - algebraMap R (End R M) μ).restrict h) := by
  apply isNilpotent_restrict_of_le (q := f.genEigenspace μ (maxUnifEigenspaceIndex f μ))
    _ (isNilpotent_restrict_genEigenspace_nat f μ (maxUnifEigenspaceIndex f μ))
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f ((algebraMap R (Module.End R M)) μ)) ↑( …
    ⊢ LE.le (iSup fun k => (f.genEigenspace μ) ↑k) ((f.genEigenspace μ) ↑(f.maxUni …
  -/
  apply iSup_le
  /-
    case h
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f ((algebraMap R (Module.End R M)) μ)) ↑( …
    ⊢ ∀ (i : Nat), LE.le ((f.genEigenspace μ) ↑i) ((f.genEigenspace μ) ↑(f.maxUnif …
  -/
  intro k
  /-
    case h
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Module.End R M
    μ : R
    h : optParam (Set.MapsTo ⇑(HSub.hSub f ((algebraMap R (Module.End R M)) μ)) ↑( …
    k : Nat
    ⊢ LE.le ((f.genEigenspace μ) ↑k) ((f.genEigenspace μ) ↑(f.maxUnifEigenspaceInd …
  -/
  apply genEigenspace_le_genEigenspace_maxUnifEigenspaceIndex
  /-
    🎉 no goals
  -/


lemma disjoint_genEigenspace [NoZeroSMulDivisors R M]
    (f : End R M) {μ₁ μ₂ : R} (hμ : μ₁ ≠ μ₂) (k l : ℕ∞) :
    Disjoint (f.genEigenspace μ₁ k) (f.genEigenspace μ₂ l) := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k l : ENat
    ⊢ Disjoint ((f.genEigenspace μ₁) k) ((f.genEigenspace μ₂) l)
  -/
  rw [genEigenspace_eq_iSup_genEigenspace_nat, genEigenspace_eq_iSup_genEigenspace_nat]
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k l : ENat
    ⊢ Disjoint (iSup fun l => (f.genEigenspace μ₁) ↑↑l) (iSup fun l_1 => (f.genEig …
  -/
  simp_rw [genEigenspace_directed.disjoint_iSup_left, genEigenspace_directed.disjoint_iSup_right]
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k l : ENat
    ⊢ ∀ (i : Subtype fun l => LE.le (↑l) k) (i_1 : Subtype fun l_1 => LE.le (↑l_1) …
  -/
  rintro ⟨k, -⟩ ⟨l, -⟩
  /-
    case mk.mk
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    ⊢ Disjoint ((f.genEigenspace μ₁) ↑↑⟨k, property✝¹⟩) ((f.genEigenspace μ₂) ↑↑⟨l …
  -/
  nontriviality M
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    ⊢ Disjoint ((f.genEigenspace μ₁) ↑↑⟨k, property✝¹⟩) ((f.genEigenspace μ₂) ↑↑⟨l …
  -/
  have := NoZeroSMulDivisors.isReduced R M
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    this : IsReduced R
    ⊢ Disjoint ((f.genEigenspace μ₁) ↑↑⟨k, property✝¹⟩) ((f.genEigenspace μ₂) ↑↑⟨l …
  -/
  rw [disjoint_iff]
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    this : IsReduced R
    ⊢ Eq (Min.min ((f.genEigenspace μ₁) ↑↑⟨k, property✝¹⟩) ((f.genEigenspace μ₂) ↑ …
  -/
  set p := f.genEigenspace μ₁ k ⊓ f.genEigenspace μ₂ l
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    this : IsReduced R
    p : Submodule R M := Min.min ((f.genEigenspace μ₁) ↑k) ((f.genEigenspace μ₂) ↑l)
    ⊢ Eq p Bot.bot
  -/
  by_contra hp
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    this : IsReduced R
    p : Submodule R M := Min.min ((f.genEigenspace μ₁) ↑k) ((f.genEigenspace μ₂) ↑l)
    hp : Not (Eq p Bot.bot)
    ⊢ False
  -/
  replace hp : Nontrivial p := Submodule.nontrivial_iff_ne_bot.mpr hp
  let f₁ : End R p := (f - algebraMap R (End R M) μ₁).restrict <| MapsTo.inter_inter
    (mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ₁) μ₁ k)
    (mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ₁) μ₂ l)
  let f₂ : End R p := (f - algebraMap R (End R M) μ₂).restrict <| MapsTo.inter_inter
    (mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ₂) μ₁ k)
    (mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ₂) μ₂ l)
  have : IsNilpotent (f₂ - f₁) := by
    apply Commute.isNilpotent_sub (x := f₂) (y := f₁) _
      (isNilpotent_restrict_of_le inf_le_right _)
      (isNilpotent_restrict_of_le inf_le_left _)
    · ext; simp [f₁, f₂, smul_sub, sub_sub, smul_comm μ₁, add_sub_left_comm]
    apply mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f _)
    apply isNilpotent_restrict_genEigenspace_nat
    apply mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f _)
    apply isNilpotent_restrict_genEigenspace_nat
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    this✝ : IsReduced R
    p : Submodule R M := Min.min ((f.genEigenspace μ₁) ↑k) ((f.genEigenspace μ₂) ↑l)
    hp : Nontrivial (Subtype fun x => Membership.mem p x)
    f₁ : Module.End R (Subtype fun x => Membership.mem p x) := LinearMap.restrict  …
    f₂ : Module.End R (Subtype fun x => Membership.mem p x) := LinearMap.restrict  …
    this : IsNilpotent (HSub.hSub f₂ f₁)
    ⊢ False
  -/
  have hf₁₂ : f₂ - f₁ = algebraMap R (End R p) (μ₁ - μ₂) := by ext; simp [f₁, f₂, sub_smul]
  rw [hf₁₂, IsNilpotent.map_iff (NoZeroSMulDivisors.algebraMap_injective R (End R p)),
    isNilpotent_iff_eq_zero, sub_eq_zero] at this
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    k✝ l✝ : ENat
    k : Nat
    property✝¹ : LE.le (↑k) k✝
    l : Nat
    property✝ : LE.le (↑l) l✝
    a✝ : Nontrivial M
    this✝ : IsReduced R
    p : Submodule R M := Min.min ((f.genEigenspace μ₁) ↑k) ((f.genEigenspace μ₂) ↑l)
    hp : Nontrivial (Subtype fun x => Membership.mem p x)
    f₁ : Module.End R (Subtype fun x => Membership.mem p x) := LinearMap.restrict  …
    f₂ : Module.End R (Subtype fun x => Membership.mem p x) := LinearMap.restrict  …
    this : Eq μ₁ μ₂
    hf₁₂ : Eq (HSub.hSub f₂ f₁) ((algebraMap R (Module.End R (Subtype fun x => Mem …
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


lemma injOn_genEigenspace [NoZeroSMulDivisors R M] (f : End R M) (k : ℕ∞) :
    InjOn (f.genEigenspace · k) {μ | f.genEigenspace μ k ≠ ⊥} := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    k : ENat
    ⊢ Set.InjOn (fun x => (f.genEigenspace x) k) (setOf fun μ => Ne ((f.genEigensp …
  -/
  rintro μ₁ _ μ₂ hμ₂ hμ₁₂
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    k : ENat
    μ₁ : R
    a✝ : Membership.mem (setOf fun μ => Ne ((f.genEigenspace μ) k) Bot.bot) μ₁
    μ₂ : R
    hμ₂ : Membership.mem (setOf fun μ => Ne ((f.genEigenspace μ) k) Bot.bot) μ₂
    hμ₁₂ : Eq ((fun x => (f.genEigenspace x) k) μ₁) ((fun x => (f.genEigenspace x) …
    ⊢ Eq μ₁ μ₂
  -/
  by_contra contra
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    k : ENat
    μ₁ : R
    a✝ : Membership.mem (setOf fun μ => Ne ((f.genEigenspace μ) k) Bot.bot) μ₁
    μ₂ : R
    hμ₂ : Membership.mem (setOf fun μ => Ne ((f.genEigenspace μ) k) Bot.bot) μ₂
    hμ₁₂ : Eq ((fun x => (f.genEigenspace x) k) μ₁) ((fun x => (f.genEigenspace x) …
    contra : Not (Eq μ₁ μ₂)
    ⊢ False
  -/
  apply hμ₂
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    k : ENat
    μ₁ : R
    a✝ : Membership.mem (setOf fun μ => Ne ((f.genEigenspace μ) k) Bot.bot) μ₁
    μ₂ : R
    hμ₂ : Membership.mem (setOf fun μ => Ne ((f.genEigenspace μ) k) Bot.bot) μ₂
    hμ₁₂ : Eq ((fun x => (f.genEigenspace x) k) μ₁) ((fun x => (f.genEigenspace x) …
    contra : Not (Eq μ₁ μ₂)
    ⊢ Eq ((f.genEigenspace μ₂) k) Bot.bot
  -/
  simpa only [hμ₁₂, disjoint_self] using f.disjoint_genEigenspace contra k k
  /-
    🎉 no goals
  -/


@[deprecated disjoint_genEigenspace (since := "2024-10-23")]
lemma disjoint_iSup_genEigenspace [NoZeroSMulDivisors R M]
    (f : End R M) {μ₁ μ₂ : R} (hμ : μ₁ ≠ μ₂) :
    Disjoint (⨆ k : ℕ, f.genEigenspace μ₁ k) (⨆ k : ℕ, f.genEigenspace μ₂ k) := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    μ₁ μ₂ : R
    hμ : Ne μ₁ μ₂
    ⊢ Disjoint (iSup fun k => (f.genEigenspace μ₁) ↑k) (iSup fun k => (f.genEigens …
  -/
  simpa only [iSup_genEigenspace_eq] using disjoint_genEigenspace f hμ ⊤ ⊤
  /-
    🎉 no goals
  -/


lemma injOn_maxGenEigenspace [NoZeroSMulDivisors R M] (f : End R M) :
    InjOn (f.maxGenEigenspace ·) {μ | f.maxGenEigenspace μ ≠ ⊥} :=
  injOn_genEigenspace f ⊤


@[deprecated injOn_genEigenspace (since := "2024-10-23")]
lemma injOn_iSup_genEigenspace [NoZeroSMulDivisors R M] (f : End R M) :
    InjOn (⨆ k : ℕ, f.genEigenspace · k) {μ | ⨆ k : ℕ, f.genEigenspace μ k ≠ ⊥} := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    ⊢ Set.InjOn (fun x => iSup fun k => (f.genEigenspace x) ↑k) (setOf fun μ => Ne …
  -/
  simp_rw [iSup_genEigenspace_eq]
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    ⊢ Set.InjOn (fun x => f.maxGenEigenspace x) (setOf fun μ => Ne (f.maxGenEigens …
  -/
  apply injOn_maxGenEigenspace
  /-
    🎉 no goals
  -/


theorem independent_genEigenspace [NoZeroSMulDivisors R M] (f : End R M) (k : ℕ∞) :
    iSupIndep (f.genEigenspace · k) := by
  classical
  suffices ∀ μ₁ (s : Finset R), μ₁ ∉ s → Disjoint (f.genEigenspace μ₁ k)
    (s.sup fun μ ↦ f.genEigenspace μ k) by
    simp_rw [iSupIndep_iff_supIndep_of_injOn (injOn_genEigenspace f k),
      Finset.supIndep_iff_disjoint_erase]
    exact fun s μ _ ↦ this _ _ (s.not_mem_erase μ)
  intro μ₁ s
  induction' s using Finset.induction_on with μ₂ s _ ih
  · simp
  intro hμ₁₂
  obtain ⟨hμ₁₂ : μ₁ ≠ μ₂, hμ₁ : μ₁ ∉ s⟩ := by rwa [Finset.mem_insert, not_or] at hμ₁₂
  specialize ih hμ₁
  rw [Finset.sup_insert, disjoint_iff, Submodule.eq_bot_iff]
  rintro x ⟨hx, hx'⟩
  simp only [SetLike.mem_coe] at hx hx'
  suffices x ∈ genEigenspace f μ₂ k by
    rw [← Submodule.mem_bot (R := R), ← (f.disjoint_genEigenspace hμ₁₂ k k).eq_bot]
    exact ⟨hx, this⟩
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.mp hx'; clear hx'
  let g := f - μ₂ • 1
  simp_rw [mem_genEigenspace, ← exists_prop] at hy ⊢
  peel hy with l hlk hl
  simp only [mem_genEigenspace_nat, LinearMap.mem_ker] at hl
  have hyz : (g ^ l) (y + z) ∈
      (f.genEigenspace μ₁ k) ⊓ s.sup fun μ ↦ f.genEigenspace μ k := by
    refine ⟨f.mapsTo_genEigenspace_of_comm (g := g ^ l) ?_ μ₁ k hx, ?_⟩
    · exact Algebra.mul_sub_algebraMap_pow_commutes f μ₂ l
    · rw [SetLike.mem_coe, map_add, hl, zero_add]
      suffices (s.sup fun μ ↦ f.genEigenspace μ k).map (g ^ l) ≤
          s.sup fun μ ↦ f.genEigenspace μ k by exact this (Submodule.mem_map_of_mem hz)
      simp_rw [Finset.sup_eq_iSup, Submodule.map_iSup (ι := R), Submodule.map_iSup (ι := _ ∈ s)]
      refine iSup₂_mono fun μ _ ↦ ?_
      rintro - ⟨u, hu, rfl⟩
      refine f.mapsTo_genEigenspace_of_comm ?_ μ k hu
      exact Algebra.mul_sub_algebraMap_pow_commutes f μ₂ l
  rwa [ih.eq_bot, Submodule.mem_bot] at hyz


theorem independent_maxGenEigenspace [NoZeroSMulDivisors R M] (f : End R M) :
    iSupIndep f.maxGenEigenspace := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    ⊢ iSupIndep f.maxGenEigenspace
  -/
  apply independent_genEigenspace
  /-
    🎉 no goals
  -/


@[deprecated independent_genEigenspace (since := "2024-10-23")]
theorem independent_iSup_genEigenspace [NoZeroSMulDivisors R M] (f : End R M) :
    iSupIndep (fun μ ↦ ⨆ k : ℕ, f.genEigenspace μ k) := by
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    ⊢ iSupIndep fun μ => iSup fun k => (f.genEigenspace μ) ↑k
  -/
  simp_rw [iSup_genEigenspace_eq]
  /-
    R : Type v
    M : Type w
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : Module.End R M
    ⊢ iSupIndep fun μ => f.maxGenEigenspace μ
  -/
  apply independent_maxGenEigenspace
  /-
    🎉 no goals
  -/


/-- The eigenspaces of a linear operator form an independent family of subspaces of `M`.  That is,
any eigenspace has trivial intersection with the span of all the other eigenspaces. -/
theorem eigenspaces_iSupIndep [NoZeroSMulDivisors R M] (f : End R M) :
    iSupIndep f.eigenspace :=
  (f.independent_genEigenspace 1).mono fun _ ↦ le_rfl


@[deprecated (since := "2024-11-24")] alias eigenspaces_independent := eigenspaces_iSupIndep


/-- Eigenvectors corresponding to distinct eigenvalues of a linear operator are linearly
    independent. -/
theorem eigenvectors_linearIndependent' {ι : Type*} [NoZeroSMulDivisors R M]
    (f : End R M) (μ : ι → R) (hμ : Function.Injective μ) (v : ι → M)
    (h_eigenvec : ∀ i, f.HasEigenvector (μ i) (v i)) : LinearIndependent R v :=
  f.eigenspaces_iSupIndep.comp hμ |>.linearIndependent _
    (fun i ↦ h_eigenvec i |>.left) (fun i ↦ h_eigenvec i |>.right)


/-- Eigenvectors corresponding to distinct eigenvalues of a linear operator are linearly
    independent. (Lemma 5.10 of [axler2015])

    We use the eigenvalues as indexing set to ensure that there is only one eigenvector for each
    eigenvalue in the image of `xs`.
    See `Module.End.eigenvectors_linearIndependent'` for an indexed variant. -/
theorem eigenvectors_linearIndependent [NoZeroSMulDivisors R M]
    (f : End R M) (μs : Set R) (xs : μs → M)
    (h_eigenvec : ∀ μ : μs, f.HasEigenvector μ (xs μ)) : LinearIndependent R xs :=
  f.eigenvectors_linearIndependent' (fun μ : μs ↦ μ) Subtype.coe_injective _ h_eigenvec


/-- If `f` maps a subspace `p` into itself, then the generalized eigenspace of the restriction
    of `f` to `p` is the part of the generalized eigenspace of `f` that lies in `p`. -/
theorem genEigenspace_restrict (f : End R M) (p : Submodule R M) (k : ℕ∞) (μ : R)
    (hfp : ∀ x : M, x ∈ p → f x ∈ p) :
    genEigenspace (LinearMap.restrict f hfp) μ k =
      Submodule.comap p.subtype (f.genEigenspace μ k) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    k : ENat
    μ : R
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    ⊢ Eq ((Module.End.genEigenspace (LinearMap.restrict f hfp) μ) k) (Submodule.co …
  -/
  ext x
  suffices ∀ l : ℕ, genEigenspace (LinearMap.restrict f hfp) μ l =
      Submodule.comap p.subtype (f.genEigenspace μ l) by
    simp_rw [mem_genEigenspace, ← mem_genEigenspace_nat, this,
      Submodule.mem_comap, mem_genEigenspace (k := k), mem_genEigenspace_nat]
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    k : ENat
    μ : R
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    x : Subtype fun x => Membership.mem p x
    ⊢ ∀ (l : Nat), Eq ((Module.End.genEigenspace (LinearMap.restrict f hfp) μ) ↑l) …
  -/
  intro l
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    k : ENat
    μ : R
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    x : Subtype fun x => Membership.mem p x
    l : Nat
    ⊢ Eq ((Module.End.genEigenspace (LinearMap.restrict f hfp) μ) ↑l) (Submodule.c …
  -/
  simp only [genEigenspace_nat, OrderHom.coe_mk, ← LinearMap.ker_comp]
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    k : ENat
    μ : R
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    x : Subtype fun x => Membership.mem p x
    l : Nat
    ⊢ Eq (LinearMap.ker (HPow.hPow (HSub.hSub (LinearMap.restrict f hfp) (HSMul.hS …
  -/
  induction' l with l ih
    /-
      case h.zero
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      p : Submodule R M
      k : ENat
      μ : R
      hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
      x : Subtype fun x => Membership.mem p x
      ⊢ Eq (LinearMap.ker (HPow.hPow (HSub.hSub (LinearMap.restrict f hfp) (HSMul.hS …
    -/
  · rw [pow_zero, pow_zero, LinearMap.one_eq_id]
    /-
      case h.zero
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      p : Submodule R M
      k : ENat
      μ : R
      hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
      x : Subtype fun x => Membership.mem p x
      ⊢ Eq (LinearMap.ker LinearMap.id) (LinearMap.ker (LinearMap.comp 1 p.subtype))
    -/
    apply (Submodule.ker_subtype _).symm
    /-
      🎉 no goals
    -/
  · erw [pow_succ, pow_succ, LinearMap.ker_comp, LinearMap.ker_comp, ih, ← LinearMap.ker_comp,
      LinearMap.comp_assoc]


lemma _root_.Submodule.inf_genEigenspace (f : End R M) (p : Submodule R M) {k : ℕ∞} {μ : R}
    (hfp : ∀ x : M, x ∈ p → f x ∈ p) :
    p ⊓ f.genEigenspace μ k =
      (genEigenspace (LinearMap.restrict f hfp) μ k).map p.subtype := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    k : ENat
    μ : R
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    ⊢ Eq (Min.min p ((f.genEigenspace μ) k)) (Submodule.map p.subtype ((Module.End …
  -/
  rw [f.genEigenspace_restrict _ _ _ hfp, Submodule.map_comap_eq, Submodule.range_subtype]
  /-
    🎉 no goals
  -/


lemma mapsTo_restrict_maxGenEigenspace_restrict_of_mapsTo
    {p : Submodule R M} (f g : End R M) (hf : MapsTo f p p) (hg : MapsTo g p p) {μ₁ μ₂ : R}
    (h : MapsTo f (g.maxGenEigenspace μ₁) (g.maxGenEigenspace μ₂)) :
    MapsTo (f.restrict hf)
      (maxGenEigenspace (g.restrict hg) μ₁)
      (maxGenEigenspace (g.restrict hg) μ₂) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    f g : Module.End R M
    hf : Set.MapsTo ⇑f ↑p ↑p
    hg : Set.MapsTo ⇑g ↑p ↑p
    μ₁ μ₂ : R
    h : Set.MapsTo ⇑f ↑(g.maxGenEigenspace μ₁) ↑(g.maxGenEigenspace μ₂)
    ⊢ Set.MapsTo ⇑(LinearMap.restrict f hf) ↑(Module.End.maxGenEigenspace (LinearM …
  -/
  intro x hx
  simp_rw [SetLike.mem_coe, mem_maxGenEigenspace, ← LinearMap.restrict_smul_one _,
    LinearMap.restrict_sub _, LinearMap.pow_restrict _, LinearMap.restrict_apply,
    Submodule.mk_eq_zero, ← mem_maxGenEigenspace] at hx ⊢
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    f g : Module.End R M
    hf : Set.MapsTo ⇑f ↑p ↑p
    hg : Set.MapsTo ⇑g ↑p ↑p
    μ₁ μ₂ : R
    h : Set.MapsTo ⇑f ↑(g.maxGenEigenspace μ₁) ↑(g.maxGenEigenspace μ₂)
    x : Subtype fun x => Membership.mem p x
    hx : Membership.mem (g.maxGenEigenspace μ₁) ↑x
    ⊢ Membership.mem (g.maxGenEigenspace μ₂) (f ↑x)
  -/
  exact h hx
  /-
    🎉 no goals
  -/


/-- If `p` is an invariant submodule of an endomorphism `f`, then the `μ`-eigenspace of the
restriction of `f` to `p` is a submodule of the `μ`-eigenspace of `f`. -/
theorem eigenspace_restrict_le_eigenspace (f : End R M) {p : Submodule R M} (hfp : ∀ x ∈ p, f x ∈ p)
    (μ : R) : (eigenspace (f.restrict hfp) μ).map p.subtype ≤ f.eigenspace μ := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    μ : R
    ⊢ LE.le (Submodule.map p.subtype (Module.End.eigenspace (LinearMap.restrict f  …
  -/
  rintro a ⟨x, hx, rfl⟩
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    μ : R
    x : Subtype fun x => Membership.mem p x
    hx : Membership.mem (↑(Module.End.eigenspace (LinearMap.restrict f hfp) μ)) x
    ⊢ Membership.mem (f.eigenspace μ) (p.subtype x)
  -/
  simp only [SetLike.mem_coe, mem_eigenspace_iff, LinearMap.restrict_apply] at hx ⊢
  /-
    case intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    μ : R
    x : Subtype fun x => Membership.mem p x
    hx : Eq ⟨f ↑x, ⋯⟩ (HSMul.hSMul μ x)
    ⊢ Eq (f (p.subtype x)) (HSMul.hSMul μ (p.subtype x))
  -/
  exact congr_arg Subtype.val hx
  /-
    🎉 no goals
  -/


/-- Generalized eigenrange and generalized eigenspace for exponent `finrank K V` are disjoint. -/
theorem generalized_eigenvec_disjoint_range_ker [FiniteDimensional K V] (f : End K V) (μ : K) :
    Disjoint (f.genEigenrange μ (finrank K V))
      (f.genEigenspace μ (finrank K V)) := by
  have h :=
    calc
      Submodule.comap ((f - μ • 1) ^ finrank K V)
        (f.genEigenspace μ (finrank K V)) =
          LinearMap.ker ((f - algebraMap _ _ μ) ^ finrank K V *
            (f - algebraMap K (End K V) μ) ^ finrank K V) := by
              rw [genEigenspace_nat, ← LinearMap.ker_comp]; rfl
      _ = f.genEigenspace μ (finrank K V + finrank K V : ℕ) := by
              simp_rw [← pow_add, genEigenspace_nat]; rfl
      _ = f.genEigenspace μ (finrank K V) := by
              rw [genEigenspace_eq_genEigenspace_finrank_of_le]; omega
  rw [disjoint_iff_inf_le, genEigenrange_nat, LinearMap.range_eq_map,
    Submodule.map_inf_eq_map_inf_comap, top_inf_eq, h, genEigenspace_nat]
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : Eq (Submodule.comap (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) (Module.fin …
    ⊢ LE.le (Submodule.map (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) (Module.finr …
  -/
  apply Submodule.map_comap_le
  /-
    🎉 no goals
  -/


/-- If an invariant subspace `p` of an endomorphism `f` is disjoint from the `μ`-eigenspace of `f`,
then the restriction of `f` to `p` has trivial `μ`-eigenspace. -/
theorem eigenspace_restrict_eq_bot {f : End R M} {p : Submodule R M} (hfp : ∀ x ∈ p, f x ∈ p)
    {μ : R} (hμp : Disjoint (f.eigenspace μ) p) : eigenspace (f.restrict hfp) μ = ⊥ := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    μ : R
    hμp : Disjoint (f.eigenspace μ) p
    ⊢ Eq (Module.End.eigenspace (LinearMap.restrict f hfp) μ) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    μ : R
    hμp : Disjoint (f.eigenspace μ) p
    ⊢ LE.le (Module.End.eigenspace (LinearMap.restrict f hfp) μ) Bot.bot
  -/
  intro x hx
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hfp : ∀ (x : M), Membership.mem p x → Membership.mem p (f x)
    μ : R
    hμp : Disjoint (f.eigenspace μ) p
    x : Subtype fun x => Membership.mem p x
    hx : Membership.mem (Module.End.eigenspace (LinearMap.restrict f hfp) μ) x
    ⊢ Membership.mem Bot.bot x
  -/
  simpa using hμp.le_bot ⟨eigenspace_restrict_le_eigenspace f hfp μ ⟨x, hx, rfl⟩, x.prop⟩
  /-
    🎉 no goals
  -/


/-- The generalized eigenspace of an eigenvalue has positive dimension for positive exponents. -/
theorem pos_finrank_genEigenspace_of_hasEigenvalue [FiniteDimensional K V] {f : End K V}
    {k : ℕ} {μ : K} (hx : f.HasEigenvalue μ) (hk : 0 < k) :
    0 < finrank K (f.genEigenspace μ k) :=
  calc
                                            /-
                                              K : Type v
                                              V : Type w
                                              inst✝³ : Field K
                                              inst✝² : AddCommGroup V
                                              inst✝¹ : Module K V
                                              inst✝ : FiniteDimensional K V
                                              f : Module.End K V
                                              k : Nat
                                              μ : K
                                              hx : f.HasEigenvalue μ
                                              hk : LT.lt 0 k
                                              ⊢ Eq 0 (Module.finrank K (Subtype fun x => Membership.mem Bot.bot x))
                                            -/
    0 = finrank K (⊥ : Submodule K V) := by rw [finrank_bot]
                                            /-
                                              🎉 no goals
                                            -/
    _ < finrank K (f.eigenspace μ) := Submodule.finrank_lt_finrank_of_lt (bot_lt_iff_ne_bot.2 hx)
    _ ≤ finrank K (f.genEigenspace μ k) :=
                                                               /-
                                                                 K : Type v
                                                                 V : Type w
                                                                 inst✝³ : Field K
                                                                 inst✝² : AddCommGroup V
                                                                 inst✝¹ : Module K V
                                                                 inst✝ : FiniteDimensional K V
                                                                 f : Module.End K V
                                                                 k : Nat
                                                                 μ : K
                                                                 hx : f.HasEigenvalue μ
                                                                 hk : LT.lt 0 k
                                                                 ⊢ LE.le 1 ↑k
                                                               -/
      Submodule.finrank_mono ((f.genEigenspace μ).monotone (by simpa using Nat.succ_le_of_lt hk))
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A linear map maps a generalized eigenrange into itself. -/
theorem map_genEigenrange_le {f : End K V} {μ : K} {n : ℕ} :
    Submodule.map f (f.genEigenrange μ n) ≤ f.genEigenrange μ n :=
  calc
    Submodule.map f (f.genEigenrange μ n) =
      LinearMap.range (f * (f - algebraMap _ _ μ) ^ n) := by
        /-
          K : Type v
          V : Type w
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          f : Module.End K V
          μ : K
          n : Nat
          ⊢ Eq (Submodule.map f (f.genEigenrange μ ↑n)) (LinearMap.range (HMul.hMul f (H …
        -/
        rw [genEigenrange_nat]; exact (LinearMap.range_comp _ _).symm
                                /-
                                  🎉 no goals
                                -/
    _ = LinearMap.range ((f - algebraMap _ _ μ) ^ n * f) := by
        /-
          K : Type v
          V : Type w
          inst✝² : Field K
          inst✝¹ : AddCommGroup V
          inst✝ : Module K V
          f : Module.End K V
          μ : K
          n : Nat
          ⊢ Eq (LinearMap.range (HMul.hMul f (HPow.hPow (HSub.hSub f ((algebraMap K (Mod …
        -/
        rw [Algebra.mul_sub_algebraMap_pow_commutes]
        /-
          🎉 no goals
        -/
    _ = Submodule.map ((f - algebraMap _ _ μ) ^ n) (LinearMap.range f) := LinearMap.range_comp _ _
                                  /-
                                    K : Type v
                                    V : Type w
                                    inst✝² : Field K
                                    inst✝¹ : AddCommGroup V
                                    inst✝ : Module K V
                                    f : Module.End K V
                                    μ : K
                                    n : Nat
                                    ⊢ LE.le (Submodule.map (HPow.hPow (HSub.hSub f ((algebraMap K (Module.End K V) …
                                  -/
    _ ≤ f.genEigenrange μ n := by rw [genEigenrange_nat]; apply LinearMap.map_le_range
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma genEigenspace_le_smul (f : Module.End R M) (μ t : R) (k : ℕ∞) :
    (f.genEigenspace μ k) ≤ (t • f).genEigenspace (t * μ) k := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ t : R
    k : ENat
    ⊢ LE.le ((f.genEigenspace μ) k) (((HSMul.hSMul t f).genEigenspace (HMul.hMul t …
  -/
  intro m hm
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ t : R
    k : ENat
    m : M
    hm : Membership.mem ((f.genEigenspace μ) k) m
    ⊢ Membership.mem (((HSMul.hSMul t f).genEigenspace (HMul.hMul t μ)) k) m
  -/
  simp_rw [mem_genEigenspace, ← exists_prop, LinearMap.mem_ker] at hm ⊢
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ t : R
    k : ENat
    m : M
    hm : Exists fun l => Exists fun h => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul  …
    ⊢ Exists fun l => Exists fun h => Eq ((HPow.hPow (HSub.hSub (HSMul.hSMul t f)  …
  -/
  peel hm with l hlk hl
  /-
    case h.h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ t : R
    k : ENat
    m : M
    hm : Exists fun l => Exists fun h => Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul  …
    l : Nat
    hlk : LE.le (↑l) k
    hl : Eq ((HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1)) l) m) 0
    ⊢ Eq ((HPow.hPow (HSub.hSub (HSMul.hSMul t f) (HSMul.hSMul (HMul.hMul t μ) 1)) …
  -/
  rw [mul_smul, ← smul_sub, smul_pow, LinearMap.smul_apply, hl, smul_zero]
  /-
    🎉 no goals
  -/


@[deprecated genEigenspace_le_smul (since := "2024-10-23")]
lemma iSup_genEigenspace_le_smul (f : Module.End R M) (μ t : R) :
    (⨆ k : ℕ, f.genEigenspace μ k) ≤ ⨆ k : ℕ, (t • f).genEigenspace (t * μ) k := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ t : R
    ⊢ LE.le (iSup fun k => (f.genEigenspace μ) ↑k) (iSup fun k => ((HSMul.hSMul t  …
  -/
  rw [iSup_genEigenspace_eq, iSup_genEigenspace_eq]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    μ t : R
    ⊢ LE.le (f.maxGenEigenspace μ) ((HSMul.hSMul t f).maxGenEigenspace (HMul.hMul  …
  -/
  apply genEigenspace_le_smul
  /-
    🎉 no goals
  -/


lemma genEigenspace_inf_le_add
    (f₁ f₂ : End R M) (μ₁ μ₂ : R) (k₁ k₂ : ℕ∞) (h : Commute f₁ f₂) :
    (f₁.genEigenspace μ₁ k₁) ⊓ (f₂.genEigenspace μ₂ k₂) ≤
    (f₁ + f₂).genEigenspace (μ₁ + μ₂) (k₁ + k₂) := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    h : Commute f₁ f₂
    ⊢ LE.le (Min.min ((f₁.genEigenspace μ₁) k₁) ((f₂.genEigenspace μ₂) k₂)) (((HAd …
  -/
  intro m hm
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    h : Commute f₁ f₂
    m : M
    hm : Membership.mem (Min.min ((f₁.genEigenspace μ₁) k₁) ((f₂.genEigenspace μ₂) …
    ⊢ Membership.mem (((HAdd.hAdd f₁ f₂).genEigenspace (HAdd.hAdd μ₁ μ₂)) (HAdd.hA …
  -/
  simp only [Submodule.mem_inf, mem_genEigenspace, LinearMap.mem_ker] at hm ⊢
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    h : Commute f₁ f₂
    m : M
    hm : And (Exists fun l => And (LE.le (↑l) k₁) (Eq ((HPow.hPow (HSub.hSub f₁ (H …
    ⊢ Exists fun l => And (LE.le (↑l) (HAdd.hAdd k₁ k₂)) (Eq ((HPow.hPow (HSub.hSu …
  -/
  obtain ⟨⟨l₁, hlk₁, hl₁⟩, ⟨l₂, hlk₂, hl₂⟩⟩ := hm
  /-
    case intro.intro.intro.intro.intro
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    h : Commute f₁ f₂
    m : M
    l₁ : Nat
    hlk₁ : LE.le (↑l₁) k₁
    hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
    l₂ : Nat
    hlk₂ : LE.le (↑l₂) k₂
    hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
    ⊢ Exists fun l => And (LE.le (↑l) (HAdd.hAdd k₁ k₂)) (Eq ((HPow.hPow (HSub.hSu …
  -/
  use l₁ + l₂
  have : f₁ + f₂ - (μ₁ + μ₂) • 1 = (f₁ - μ₁ • 1) + (f₂ - μ₂ • 1) := by
    rw [add_smul]; exact add_sub_add_comm f₁ f₂ (μ₁ • 1) (μ₂ • 1)
  replace h : Commute (f₁ - μ₁ • 1) (f₂ - μ₂ • 1) :=
    (h.sub_right <| Algebra.commute_algebraMap_right μ₂ f₁).sub_left
      (Algebra.commute_algebraMap_left μ₁ _)
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    m : M
    l₁ : Nat
    hlk₁ : LE.le (↑l₁) k₁
    hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
    l₂ : Nat
    hlk₂ : LE.le (↑l₂) k₂
    hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
    this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
    h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
    ⊢ And (LE.le (↑(HAdd.hAdd l₁ l₂)) (HAdd.hAdd k₁ k₂)) (Eq ((HPow.hPow (HSub.hSu …
  -/
  rw [this, h.add_pow', LinearMap.coeFn_sum, Finset.sum_apply]
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    m : M
    l₁ : Nat
    hlk₁ : LE.le (↑l₁) k₁
    hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
    l₂ : Nat
    hlk₂ : LE.le (↑l₂) k₂
    hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
    this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
    h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
    ⊢ And (LE.le (↑(HAdd.hAdd l₁ l₂)) (HAdd.hAdd k₁ k₂)) (Eq ((Finset.HasAntidiago …
  -/
  constructor
    /-
      case h.left
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f₁ f₂ : Module.End R M
      μ₁ μ₂ : R
      k₁ k₂ : ENat
      m : M
      l₁ : Nat
      hlk₁ : LE.le (↑l₁) k₁
      hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
      l₂ : Nat
      hlk₂ : LE.le (↑l₂) k₂
      hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
      this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
      h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
      ⊢ LE.le (↑(HAdd.hAdd l₁ l₂)) (HAdd.hAdd k₁ k₂)
    -/
  · simpa only [Nat.cast_add] using add_le_add hlk₁ hlk₂
    /-
      🎉 no goals
    -/
  /-
    case h.right
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    m : M
    l₁ : Nat
    hlk₁ : LE.le (↑l₁) k₁
    hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
    l₂ : Nat
    hlk₂ : LE.le (↑l₂) k₂
    hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
    this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
    h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd l₁ l₂)).sum fun c => (HS …
  -/
  refine Finset.sum_eq_zero fun ⟨i, j⟩ hij ↦ ?_
  suffices (((f₁ - μ₁ • 1) ^ i) * ((f₂ - μ₂ • 1) ^ j)) m = 0 by
    rw [LinearMap.smul_apply, this, smul_zero]
  /-
    case h.right
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    m : M
    l₁ : Nat
    hlk₁ : LE.le (↑l₁) k₁
    hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
    l₂ : Nat
    hlk₂ : LE.le (↑l₂) k₂
    hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
    this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
    h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
    x✝ : Prod Nat Nat
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd l₁ l₂)) { …
    ⊢ Eq ((HMul.hMul (HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) i) (HPow.hPow (H …
  -/
  rw [Finset.mem_antidiagonal] at hij
  /-
    case h.right
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    k₁ k₂ : ENat
    m : M
    l₁ : Nat
    hlk₁ : LE.le (↑l₁) k₁
    hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
    l₂ : Nat
    hlk₂ : LE.le (↑l₂) k₂
    hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
    this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
    h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
    x✝ : Prod Nat Nat
    i j : Nat
    hij : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.h …
    ⊢ Eq ((HMul.hMul (HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) i) (HPow.hPow (H …
  -/
  obtain hi|hj : l₁ ≤ i ∨ l₂ ≤ j := by omega
  · rw [(h.pow_pow i j).eq, LinearMap.mul_apply, LinearMap.pow_map_zero_of_le hi hl₁,
      LinearMap.map_zero]
    /-
      case h.right.inr
      R : Type v
      M : Type w
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f₁ f₂ : Module.End R M
      μ₁ μ₂ : R
      k₁ k₂ : ENat
      m : M
      l₁ : Nat
      hlk₁ : LE.le (↑l₁) k₁
      hl₁ : Eq ((HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) l₁) m) 0
      l₂ : Nat
      hlk₂ : LE.le (↑l₂) k₂
      hl₂ : Eq ((HPow.hPow (HSub.hSub f₂ (HSMul.hSMul μ₂ 1)) l₂) m) 0
      this : Eq (HSub.hSub (HAdd.hAdd f₁ f₂) (HSMul.hSMul (HAdd.hAdd μ₁ μ₂) 1)) (HAd …
      h : Commute (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) (HSub.hSub f₂ (HSMul.hSMul μ₂ 1))
      x✝ : Prod Nat Nat
      i j : Nat
      hij : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.h …
      hj : LE.le l₂ j
      ⊢ Eq ((HMul.hMul (HPow.hPow (HSub.hSub f₁ (HSMul.hSMul μ₁ 1)) i) (HPow.hPow (H …
    -/
  · rw [LinearMap.mul_apply, LinearMap.pow_map_zero_of_le hj hl₂, LinearMap.map_zero]
    /-
      🎉 no goals
    -/


@[deprecated genEigenspace_inf_le_add (since := "2024-10-23")]
lemma iSup_genEigenspace_inf_le_add
    (f₁ f₂ : End R M) (μ₁ μ₂ : R) (h : Commute f₁ f₂) :
    (⨆ k : ℕ, f₁.genEigenspace μ₁ k) ⊓ (⨆ k : ℕ, f₂.genEigenspace μ₂ k) ≤
    ⨆ k : ℕ, (f₁ + f₂).genEigenspace (μ₁ + μ₂) k := by
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    h : Commute f₁ f₂
    ⊢ LE.le (Min.min (iSup fun k => (f₁.genEigenspace μ₁) ↑k) (iSup fun k => (f₂.g …
  -/
  simp_rw [iSup_genEigenspace_eq]
  /-
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    h : Commute f₁ f₂
    ⊢ LE.le (Min.min (f₁.maxGenEigenspace μ₁) (f₂.maxGenEigenspace μ₂)) ((HAdd.hAd …
  -/
  apply genEigenspace_inf_le_add
  /-
    case h
    R : Type v
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f₁ f₂ : Module.End R M
    μ₁ μ₂ : R
    h : Commute f₁ f₂
    ⊢ Commute f₁ f₂
  -/
  assumption
  /-
    🎉 no goals
  -/


lemma map_smul_of_iInf_genEigenspace_ne_bot [NoZeroSMulDivisors R M]
    {L F : Type*} [SMul R L] [FunLike F L (End R M)] [MulActionHomClass F R L (End R M)] (f : F)
    (μ : L → R) (k : ℕ∞) (h_ne : ⨅ x, (f x).genEigenspace (μ x) k ≠ ⊥)
    (t : R) (x : L) :
    μ (t • x) = t • μ x := by
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    ⊢ Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x))
  -/
  by_contra contra
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    contra : Not (Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x)))
    ⊢ False
  -/
  let g : L → Submodule R M := fun x ↦ (f x).genEigenspace (μ x) k
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    contra : Not (Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    ⊢ False
  -/
  have : ⨅ x, g x ≤ g x ⊓ g (t • x) := le_inf_iff.mpr ⟨iInf_le g x, iInf_le g (t • x)⟩
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    contra : Not (Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (g x) (g (HSMul.hSMul t x)))
    ⊢ False
  -/
  refine h_ne <| eq_bot_iff.mpr (le_trans this (disjoint_iff_inf_le.mp ?_))
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    contra : Not (Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (g x) (g (HSMul.hSMul t x)))
    ⊢ Disjoint (g x) (g (HSMul.hSMul t x))
  -/
  apply Disjoint.mono_left (genEigenspace_le_smul (f x) (μ x) t k)
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    contra : Not (Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (g x) (g (HSMul.hSMul t x)))
    ⊢ Disjoint (((HSMul.hSMul t (f x)).genEigenspace (HMul.hMul t (μ x))) k) (g (H …
  -/
  simp only [g, map_smul]
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    t : R
    x : L
    contra : Not (Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (g x) (g (HSMul.hSMul t x)))
    ⊢ Disjoint (((HSMul.hSMul t (f x)).genEigenspace (HMul.hMul t (μ x))) k) (((HS …
  -/
  exact disjoint_genEigenspace (t • f x) (Ne.symm contra) k k
  /-
    🎉 no goals
  -/


@[deprecated map_smul_of_iInf_genEigenspace_ne_bot (since := "2024-10-23")]
lemma map_smul_of_iInf_iSup_genEigenspace_ne_bot [NoZeroSMulDivisors R M]
    {L F : Type*} [SMul R L] [FunLike F L (End R M)] [MulActionHomClass F R L (End R M)] (f : F)
    (μ : L → R) (h_ne : ⨅ x, ⨆ k : ℕ, (f x).genEigenspace (μ x) k ≠ ⊥)
    (t : R) (x : L) :
    μ (t • x) = t • μ x := by
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    h_ne : Ne (iInf fun x => iSup fun k => ((f x).genEigenspace (μ x)) ↑k) Bot.bot
    t : R
    x : L
    ⊢ Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x))
  -/
  simp_rw [iSup_genEigenspace_eq] at h_ne
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : SMul R L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : MulActionHomClass F R L (Module.End R M)
    f : F
    μ : L → R
    t : R
    x : L
    h_ne : Ne (iInf fun x => (f x).maxGenEigenspace (μ x)) Bot.bot
    ⊢ Eq (μ (HSMul.hSMul t x)) (HSMul.hSMul t (μ x))
  -/
  apply map_smul_of_iInf_genEigenspace_ne_bot f μ ⊤ h_ne t x
  /-
    🎉 no goals
  -/


lemma map_add_of_iInf_genEigenspace_ne_bot_of_commute [NoZeroSMulDivisors R M]
    {L F : Type*} [Add L] [FunLike F L (End R M)] [AddHomClass F L (End R M)] (f : F)
    (μ : L → R) (k : ℕ∞) (h_ne : ⨅ x, (f x).genEigenspace (μ x) k ≠ ⊥)
    (h : ∀ x y, Commute (f x) (f y)) (x y : L) :
    μ (x + y) = μ x + μ y := by
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    ⊢ Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y))
  -/
  by_contra contra
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    contra : Not (Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y)))
    ⊢ False
  -/
  let g : L → Submodule R M := fun x ↦ (f x).genEigenspace (μ x) k
  have : ⨅ x, g x ≤ (g x ⊓ g y) ⊓ g (x + y) :=
    le_inf_iff.mpr ⟨le_inf_iff.mpr ⟨iInf_le g x, iInf_le g y⟩, iInf_le g (x + y)⟩
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    contra : Not (Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (Min.min (g x) (g y)) (g (HAdd.hAdd  …
    ⊢ False
  -/
  refine h_ne <| eq_bot_iff.mpr (le_trans this (disjoint_iff_inf_le.mp ?_))
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    contra : Not (Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (Min.min (g x) (g y)) (g (HAdd.hAdd  …
    ⊢ Disjoint (Min.min (g x) (g y)) (g (HAdd.hAdd x y))
  -/
  apply Disjoint.mono_left (genEigenspace_inf_le_add (f x) (f y) (μ x) (μ y) k k (h x y))
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    contra : Not (Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (Min.min (g x) (g y)) (g (HAdd.hAdd  …
    ⊢ Disjoint (((HAdd.hAdd (f x) (f y)).genEigenspace (HAdd.hAdd (μ x) (μ y))) (H …
  -/
  simp only [g, map_add]
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    k : ENat
    h_ne : Ne (iInf fun x => ((f x).genEigenspace (μ x)) k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    contra : Not (Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y)))
    g : L → Submodule R M := fun x => ((f x).genEigenspace (μ x)) k
    this : LE.le (iInf fun x => g x) (Min.min (Min.min (g x) (g y)) (g (HAdd.hAdd  …
    ⊢ Disjoint (((HAdd.hAdd (f x) (f y)).genEigenspace (HAdd.hAdd (μ x) (μ y))) (H …
  -/
  exact disjoint_genEigenspace (f x + f y) (Ne.symm contra) _ k
  /-
    🎉 no goals
  -/


@[deprecated map_add_of_iInf_genEigenspace_ne_bot_of_commute (since := "2024-10-23")]
lemma map_add_of_iInf_iSup_genEigenspace_ne_bot_of_commute [NoZeroSMulDivisors R M]
    {L F : Type*} [Add L] [FunLike F L (End R M)] [AddHomClass F L (End R M)] (f : F)
    (μ : L → R) (h_ne : ⨅ x, ⨆ k : ℕ, (f x).genEigenspace (μ x) k ≠ ⊥)
    (h : ∀ x y, Commute (f x) (f y)) (x y : L) :
    μ (x + y) = μ x + μ y := by
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    h_ne : Ne (iInf fun x => iSup fun k => ((f x).genEigenspace (μ x)) ↑k) Bot.bot
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    ⊢ Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y))
  -/
  simp_rw [iSup_genEigenspace_eq] at h_ne
  /-
    R : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : NoZeroSMulDivisors R M
    L : Type u_1
    F : Type u_2
    inst✝² : Add L
    inst✝¹ : FunLike F L (Module.End R M)
    inst✝ : AddHomClass F L (Module.End R M)
    f : F
    μ : L → R
    h : ∀ (x y : L), Commute (f x) (f y)
    x y : L
    h_ne : Ne (iInf fun x => (f x).maxGenEigenspace (μ x)) Bot.bot
    ⊢ Eq (μ (HAdd.hAdd x y)) (HAdd.hAdd (μ x) (μ y))
  -/
  apply map_add_of_iInf_genEigenspace_ne_bot_of_commute f μ ⊤ h_ne h x y
  /-
    🎉 no goals
  -/


