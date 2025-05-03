/-- The even or odd submodule, defined as the supremum of the even or odd powers of
`(ι Q).range`. `evenOdd 0` is the even submodule, and `evenOdd 1` is the odd submodule. -/
def evenOdd (i : ZMod 2) : Submodule R (CliffordAlgebra Q) :=
  ⨆ j : { n : ℕ // ↑n = i }, LinearMap.range (ι Q) ^ (j : ℕ)


theorem one_le_evenOdd_zero : 1 ≤ evenOdd Q 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ LE.le 1 (CliffordAlgebra.evenOdd Q 0)
  -/
  refine le_trans ?_ (le_iSup _ ⟨0, Nat.cast_zero⟩)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ LE.le 1 (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨0, ⋯⟩)
  -/
  exact (pow_zero _).ge
  /-
    🎉 no goals
  -/


theorem range_ι_le_evenOdd_one : LinearMap.range (ι Q) ≤ evenOdd Q 1 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ LE.le (LinearMap.range (CliffordAlgebra.ι Q)) (CliffordAlgebra.evenOdd Q 1)
  -/
  refine le_trans ?_ (le_iSup _ ⟨1, Nat.cast_one⟩)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ LE.le (LinearMap.range (CliffordAlgebra.ι Q)) (HPow.hPow (LinearMap.range (C …
  -/
  exact (pow_one _).ge
  /-
    🎉 no goals
  -/


theorem ι_mem_evenOdd_one (m : M) : ι Q m ∈ evenOdd Q 1 :=
  range_ι_le_evenOdd_one Q <| LinearMap.mem_range_self _ m


theorem ι_mul_ι_mem_evenOdd_zero (m₁ m₂ : M) : ι Q m₁ * ι Q m₂ ∈ evenOdd Q 0 :=
  Submodule.mem_iSup_of_mem ⟨2, rfl⟩
    (by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        m₁ m₂ : M
        ⊢ Membership.mem (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨2, ⋯⟩) ( …
      -/
      rw [Subtype.coe_mk, pow_two]
      exact
        Submodule.mul_mem_mul (LinearMap.mem_range_self (ι Q) m₁)
          (LinearMap.mem_range_self (ι Q) m₂))


theorem evenOdd_mul_le (i j : ZMod 2) : evenOdd Q i * evenOdd Q j ≤ evenOdd Q (i + j) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    i j : ZMod 2
    ⊢ LE.le (HMul.hMul (CliffordAlgebra.evenOdd Q i) (CliffordAlgebra.evenOdd Q j) …
  -/
  simp_rw [evenOdd, Submodule.iSup_eq_span, Submodule.span_mul_span]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    i j : ZMod 2
    ⊢ LE.le (Submodule.span R (HMul.hMul (Set.iUnion fun i_1 => ↑(HPow.hPow (Linea …
  -/
  apply Submodule.span_mono
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    i j : ZMod 2
    ⊢ HasSubset.Subset (HMul.hMul (Set.iUnion fun i_1 => ↑(HPow.hPow (LinearMap.ra …
  -/
  simp_rw [Set.iUnion_mul, Set.mul_iUnion, Set.iUnion_subset_iff, Set.mul_subset_iff]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    i j : ZMod 2
    ⊢ ∀ (i_1 : Subtype fun n => Eq (↑n) i) (i_2 : Subtype fun n => Eq (↑n) j) (x : …
  -/
  rintro ⟨xi, rfl⟩ ⟨yi, rfl⟩ x hx y hy
  /-
    case h.mk.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    xi yi : Nat
    x : CliffordAlgebra Q
    hx : Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨xi, …
    y : CliffordAlgebra Q
    hy : Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨yi, …
    ⊢ Membership.mem (Set.iUnion fun i => ↑(HPow.hPow (LinearMap.range (CliffordAl …
  -/
  refine Set.mem_iUnion.mpr ⟨⟨xi + yi, Nat.cast_add _ _⟩, ?_⟩
  /-
    case h.mk.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    xi yi : Nat
    x : CliffordAlgebra Q
    hx : Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨xi, …
    y : CliffordAlgebra Q
    hy : Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨yi, …
    ⊢ Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨HAdd.h …
  -/
  simp only [Subtype.coe_mk, Nat.cast_add, pow_add]
  /-
    case h.mk.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    xi yi : Nat
    x : CliffordAlgebra Q
    hx : Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨xi, …
    y : CliffordAlgebra Q
    hy : Membership.mem (↑(HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) ↑⟨yi, …
    ⊢ Membership.mem (↑(HMul.hMul (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q …
  -/
  exact Submodule.mul_mem_mul hx hy
  /-
    🎉 no goals
  -/


instance evenOdd.gradedMonoid : SetLike.GradedMonoid (evenOdd Q) where
  one_mem := Submodule.one_le.mp (one_le_evenOdd_zero Q)
  mul_mem _i _j _p _q hp hq := Submodule.mul_le.mp (evenOdd_mul_le Q _ _) _ hp _ hq


/-- A version of `CliffordAlgebra.ι` that maps directly into the graded structure. This is
primarily an auxiliary construction used to provide `CliffordAlgebra.gradedAlgebra`. -/
protected def GradedAlgebra.ι : M →ₗ[R] ⨁ i : ZMod 2, evenOdd Q i :=
  DirectSum.lof R (ZMod 2) (fun i => ↥(evenOdd Q i)) 1 ∘ₗ (ι Q).codRestrict _ (ι_mem_evenOdd_one Q)


theorem GradedAlgebra.ι_apply (m : M) :
    GradedAlgebra.ι Q m = DirectSum.of (fun i => ↥(evenOdd Q i)) 1 ⟨ι Q m, ι_mem_evenOdd_one Q m⟩ :=
  rfl


nonrec theorem GradedAlgebra.ι_sq_scalar (m : M) :
    GradedAlgebra.ι Q m * GradedAlgebra.ι Q m = algebraMap R _ (Q m) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    ⊢ Eq (HMul.hMul ((CliffordAlgebra.GradedAlgebra.ι Q) m) ((CliffordAlgebra.Grad …
  -/
  rw [GradedAlgebra.ι_apply Q, DirectSum.of_mul_of, DirectSum.algebraMap_apply]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    m : M
    ⊢ Eq ((DirectSum.of (fun i => Subtype fun x => Membership.mem (CliffordAlgebra …
  -/
  exact DirectSum.of_eq_of_gradedMonoid_eq (Sigma.subtype_ext rfl <| ι_sq_scalar _ _)
  /-
    🎉 no goals
  -/


theorem GradedAlgebra.lift_ι_eq (i' : ZMod 2) (x' : evenOdd Q i') :
    -- Porting note: added a second `by apply`
               /-
                 R : Type u_1
                 M : Type u_2
                 inst✝² : CommRing R
                 inst✝¹ : AddCommGroup M
                 inst✝ : Module R M
                 Q : QuadraticForm R M
                 i' : ZMod 2
                 x' : Subtype fun x => Membership.mem (CliffordAlgebra.evenOdd Q i') x
                 ⊢ LinearMap (RingHom.id R) M (DirectSum (ZMod 2) fun i => Subtype fun x => Mem …
               -/
               /-
                 🎉 no goals
               -/
    lift Q ⟨by apply GradedAlgebra.ι Q, by apply GradedAlgebra.ι_sq_scalar Q⟩ x' =
                                           /-
                                             🎉 no goals
                                           -/
      DirectSum.of (fun i => evenOdd Q i) i' x' := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    i' : ZMod 2
    x' : Subtype fun x => Membership.mem (CliffordAlgebra.evenOdd Q i') x
    ⊢ Eq (((CliffordAlgebra.lift Q) ⟨CliffordAlgebra.GradedAlgebra.ι Q, ⋯⟩) ↑x') ( …
  -/
  cases' x' with x' hx'
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    i' : ZMod 2
    x' : CliffordAlgebra Q
    hx' : Membership.mem (CliffordAlgebra.evenOdd Q i') x'
    ⊢ Eq (((CliffordAlgebra.lift Q) ⟨CliffordAlgebra.GradedAlgebra.ι Q, ⋯⟩) ↑⟨x',  …
  -/
  dsimp only [Subtype.coe_mk, DirectSum.lof_eq_of]
  induction hx' using Submodule.iSup_induction' with
  | mem i x hx =>
    obtain ⟨i, rfl⟩ := i
    dsimp only [Subtype.coe_mk] at hx
    induction hx using Submodule.pow_induction_on_left' with
    | algebraMap r =>
      rw [AlgHom.commutes, DirectSum.algebraMap_apply]; rfl
    | add x y i hx hy ihx ihy =>
      rw [map_add, ihx, ihy, ← AddMonoidHom.map_add]
      rfl
    | mem_mul m hm i x hx ih =>
      obtain ⟨_, rfl⟩ := hm
      rw [map_mul, ih, lift_ι_apply, GradedAlgebra.ι_apply Q, DirectSum.of_mul_of]
      refine DirectSum.of_eq_of_gradedMonoid_eq (Sigma.subtype_ext ?_ ?_) <;>
        dsimp only [GradedMonoid.mk, Subtype.coe_mk]
      · rw [Nat.succ_eq_add_one, add_comm, Nat.cast_add, Nat.cast_one]
      rfl
  | zero =>
    rw [map_zero]
    apply Eq.symm
    apply DFinsupp.single_eq_zero.mpr; rfl
  | add x y hx hy ihx ihy =>
    rw [map_add, ihx, ihy, ← AddMonoidHom.map_add]; rfl


/-- The clifford algebra is graded by the even and odd parts. -/
instance gradedAlgebra : GradedAlgebra (evenOdd Q) :=
  GradedAlgebra.ofAlgHom (evenOdd Q)
    -- while not necessary, the `by apply` makes this elaborate faster
                /-
                  R : Type u_1
                  M : Type u_2
                  inst✝² : CommRing R
                  inst✝¹ : AddCommGroup M
                  inst✝ : Module R M
                  Q : QuadraticForm R M
                  ⊢ LinearMap (RingHom.id R) M (DirectSum (ZMod 2) fun i => Subtype fun x => Mem …
                -/
                /-
                  🎉 no goals
                -/
    (lift Q ⟨by apply GradedAlgebra.ι Q, by apply GradedAlgebra.ι_sq_scalar Q⟩)
                                            /-
                                              🎉 no goals
                                            -/
    -- the proof from here onward is mostly similar to the `TensorAlgebra` case, with some extra
    -- handling for the `iSup` in `evenOdd`.
    (by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        ⊢ Eq ((DirectSum.coeAlgHom (CliffordAlgebra.evenOdd Q)).comp ((CliffordAlgebra …
      -/
      ext m
      dsimp only [LinearMap.comp_apply, AlgHom.toLinearMap_apply, AlgHom.comp_apply,
        AlgHom.id_apply]
      /-
        case a.h
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        Q : QuadraticForm R M
        m : M
        ⊢ Eq ((DirectSum.coeAlgHom (CliffordAlgebra.evenOdd Q)) (((CliffordAlgebra.lif …
      -/
      rw [lift_ι_apply, GradedAlgebra.ι_apply Q, DirectSum.coeAlgHom_of, Subtype.coe_mk])
      /-
        🎉 no goals
      -/
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          Q : QuadraticForm R M
          ⊢ ∀ (i : ZMod 2) (x : Subtype fun x => Membership.mem (CliffordAlgebra.evenOdd …
        -/
    (by apply GradedAlgebra.lift_ι_eq Q)
        /-
          🎉 no goals
        -/


theorem iSup_ι_range_eq_top : ⨆ i : ℕ, LinearMap.range (ι Q) ^ i = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ Eq (iSup fun i => HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) i) Top.top
  -/
  rw [← (DirectSum.Decomposition.isInternal (evenOdd Q)).submodule_iSup_eq_top, eq_comm]
  calc
    -- Porting note: needs extra annotations, no longer unifies against the goal in the face of
    -- ambiguity
    ⨆ (i : ZMod 2) (j : { n : ℕ // ↑n = i }), LinearMap.range (ι Q) ^ (j : ℕ) =
        ⨆ i : Σ i : ZMod 2, { n : ℕ // ↑n = i }, LinearMap.range (ι Q) ^ (i.2 : ℕ) := by
      rw [iSup_sigma]
    _ = ⨆ i : ℕ, LinearMap.range (ι Q) ^ i :=
      Function.Surjective.iSup_congr (fun i => i.2) (fun i => ⟨⟨_, i, rfl⟩, rfl⟩) fun _ => rfl


theorem evenOdd_isCompl : IsCompl (evenOdd Q 0) (evenOdd Q 1) :=
  (DirectSum.Decomposition.isInternal (evenOdd Q)).isCompl zero_ne_one <| by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      ⊢ Eq Set.univ (Insert.insert 0 (Singleton.singleton 1))
    -/
    have : (Finset.univ : Finset (ZMod 2)) = {0, 1} := rfl
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      this : Eq Finset.univ (Insert.insert 0 (Singleton.singleton 1))
      ⊢ Eq Set.univ (Insert.insert 0 (Singleton.singleton 1))
    -/
    simpa using congr_arg ((↑) : Finset (ZMod 2) → Set (ZMod 2)) this
    /-
      🎉 no goals
    -/


/-- To show a property is true on the even or odd part, it suffices to show it is true on the
scalars or vectors (respectively), closed under addition, and under left-multiplication by a pair
of vectors. -/
@[elab_as_elim]
theorem evenOdd_induction (n : ZMod 2) {motive : ∀ x, x ∈ evenOdd Q n → Prop}
    (range_ι_pow : ∀ (v) (h : v ∈ LinearMap.range (ι Q) ^ n.val),
        motive v (Submodule.mem_iSup_of_mem ⟨n.val, n.natCast_zmod_val⟩ h))
    (add : ∀ x y hx hy, motive x hx → motive y hy → motive (x + y) (Submodule.add_mem _ hx hy))
    (ι_mul_ι_mul :
      ∀ m₁ m₂ x hx,
        motive x hx →
          motive (ι Q m₁ * ι Q m₂ * x)
            (zero_add n ▸ SetLike.mul_mem_graded (ι_mul_ι_mem_evenOdd_zero Q m₁ m₂) hx))
    (x : CliffordAlgebra Q) (hx : x ∈ evenOdd Q n) : motive x hx := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q n …
    range_ι_pow : ∀ (v : CliffordAlgebra Q) (h : Membership.mem (HPow.hPow (Linear …
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q n) x
    ⊢ motive x hx
  -/
  apply Submodule.iSup_induction' (C := motive) _ _ (range_ι_pow 0 (Submodule.zero_mem _)) add
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q n …
    range_ι_pow : ∀ (v : CliffordAlgebra Q) (h : Membership.mem (HPow.hPow (Linear …
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q n) x
    ⊢ ∀ (i : Subtype fun n_1 => Eq (↑n_1) n) (x : CliffordAlgebra Q) (hx : Members …
  -/
  refine Subtype.rec ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q n …
    range_ι_pow : ∀ (v : CliffordAlgebra Q) (h : Membership.mem (HPow.hPow (Linear …
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q n) x
    ⊢ ∀ (val : Nat) (property : Eq (↑val) n) (x : CliffordAlgebra Q) (hx : Members …
  -/
  simp_rw [ZMod.natCast_eq_iff, add_comm n.val]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q n …
    range_ι_pow : ∀ (v : CliffordAlgebra Q) (h : Membership.mem (HPow.hPow (Linear …
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q n) x
    ⊢ ∀ (val : Nat) (property : Exists fun k => Eq val (HAdd.hAdd (HMul.hMul 2 k)  …
  -/
  rintro n' ⟨k, rfl⟩ xv
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q n …
    range_ι_pow : ∀ (v : CliffordAlgebra Q) (h : Membership.mem (HPow.hPow (Linear …
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q n) x
    k : Nat
    xv : CliffordAlgebra Q
    ⊢ ∀ (hx : Membership.mem (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) (H …
  -/
  simp_rw [pow_add, pow_mul]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q n …
    range_ι_pow : ∀ (v : CliffordAlgebra Q) (h : Membership.mem (HPow.hPow (Linear …
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q n) x
    k : Nat
    xv : CliffordAlgebra Q
    ⊢ ∀ (hx : Membership.mem (HMul.hMul (HPow.hPow (HPow.hPow (LinearMap.range (Cl …
  -/
  intro hxv
  induction hxv using Submodule.mul_induction_on' with
  | mem_mul_mem a ha b hb =>
    induction ha using Submodule.pow_induction_on_left' with
    | algebraMap r =>
      simp_rw [← Algebra.smul_def]
      exact range_ι_pow _ (Submodule.smul_mem _ _ hb)
    | add x y n hx hy ihx ihy =>
      simp_rw [add_mul]
      apply add _ _ _ _ ihx ihy
    | mem_mul x hx n'' y hy ihy =>
      revert hx
      simp_rw [pow_two]
      intro hx2
      induction hx2 using Submodule.mul_induction_on' with
      | mem_mul_mem m hm n hn =>
        simp_rw [LinearMap.mem_range] at hm hn
        obtain ⟨m₁, rfl⟩ := hm; obtain ⟨m₂, rfl⟩ := hn
        simp_rw [mul_assoc _ y b]
        exact ι_mul_ι_mul _ _ _ _ ihy
      | add x hx y hy ihx ihy =>
        simp_rw [add_mul]
        apply add _ _ _ _ ihx ihy
  | add x y hx hy ihx ihy =>
    apply add _ _ _ _ ihx ihy


/-- To show a property is true on the even parts, it suffices to show it is true on the
scalars, closed under addition, and under left-multiplication by a pair of vectors. -/
@[elab_as_elim]
theorem even_induction {motive : ∀ x, x ∈ evenOdd Q 0 → Prop}
    (algebraMap : ∀ r : R, motive (algebraMap _ _ r) (SetLike.algebraMap_mem_graded _ _))
    (add : ∀ x y hx hy, motive x hx → motive y hy → motive (x + y) (Submodule.add_mem _ hx hy))
    (ι_mul_ι_mul :
      ∀ m₁ m₂ x hx,
        motive x hx →
          motive (ι Q m₁ * ι Q m₂ * x)
            (zero_add (0 : ZMod 2) ▸ SetLike.mul_mem_graded (ι_mul_ι_mem_evenOdd_zero Q m₁ m₂) hx))
    (x : CliffordAlgebra Q) (hx : x ∈ evenOdd Q 0) : motive x hx := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 0 …
    algebraMap : ∀ (r : R), motive ((_root_.algebraMap R (CliffordAlgebra Q)) r) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 0) x
    ⊢ motive x hx
  -/
  refine evenOdd_induction _ _ (motive := motive) (fun rx h => ?_) add ι_mul_ι_mul x hx
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 0 …
    algebraMap : ∀ (r : R), motive ((_root_.algebraMap R (CliffordAlgebra Q)) r) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 0) x
    rx : CliffordAlgebra Q
    h : Membership.mem (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) (ZMod.va …
    ⊢ motive rx ⋯
  -/
  obtain ⟨r, rfl⟩ := Submodule.mem_one.mp h
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    motive : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 0 …
    algebraMap : ∀ (r : R), motive ((_root_.algebraMap R (CliffordAlgebra Q)) r) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 0) x
    r : R
    h : Membership.mem (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) (ZMod.va …
    ⊢ motive ((_root_.algebraMap R (CliffordAlgebra Q)) r) ⋯
  -/
  exact algebraMap r
  /-
    🎉 no goals
  -/


/-- To show a property is true on the odd parts, it suffices to show it is true on the
vectors, closed under addition, and under left-multiplication by a pair of vectors. -/
@[elab_as_elim]
theorem odd_induction {P : ∀ x, x ∈ evenOdd Q 1 → Prop}
    (ι : ∀ v, P (ι Q v) (ι_mem_evenOdd_one _ _))
    (add : ∀ x y hx hy, P x hx → P y hy → P (x + y) (Submodule.add_mem _ hx hy))
    (ι_mul_ι_mul :
      ∀ m₁ m₂ x hx,
        P x hx →
          P (CliffordAlgebra.ι Q m₁ * CliffordAlgebra.ι Q m₂ * x)
            (zero_add (1 : ZMod 2) ▸ SetLike.mul_mem_graded (ι_mul_ι_mem_evenOdd_zero Q m₁ m₂) hx))
    (x : CliffordAlgebra Q) (hx : x ∈ evenOdd Q 1) : P x hx := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 1) x → …
    ι : ∀ (v : M), P ((CliffordAlgebra.ι Q) v) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 1) x
    ⊢ P x hx
  -/
  refine evenOdd_induction _ _ (motive := P) (fun ιv => ?_) add ι_mul_ι_mul x hx
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 1) x → …
    ι : ∀ (v : M), P ((CliffordAlgebra.ι Q) v) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 1) x
    ιv : CliffordAlgebra Q
    ⊢ ∀ (h : Membership.mem (HPow.hPow (LinearMap.range (CliffordAlgebra.ι Q)) (ZM …
  -/
  simp_rw [ZMod.val_one, pow_one]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 1) x → …
    ι : ∀ (v : M), P ((CliffordAlgebra.ι Q) v) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 1) x
    ιv : CliffordAlgebra Q
    ⊢ ∀ (h : Membership.mem (LinearMap.range (CliffordAlgebra.ι Q)) ιv), P ιv ⋯
  -/
  rintro ⟨v, rfl⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : (x : CliffordAlgebra Q) → Membership.mem (CliffordAlgebra.evenOdd Q 1) x → …
    ι : ∀ (v : M), P ((CliffordAlgebra.ι Q) v) ⋯
    add : ∀ (x y : CliffordAlgebra Q) (hx : Membership.mem (CliffordAlgebra.evenOd …
    ι_mul_ι_mul : ∀ (m₁ m₂ : M) (x : CliffordAlgebra Q) (hx : Membership.mem (Clif …
    x : CliffordAlgebra Q
    hx : Membership.mem (CliffordAlgebra.evenOdd Q 1) x
    v : M
    ⊢ P ((CliffordAlgebra.ι Q) v) ⋯
  -/
  exact ι v
  /-
    🎉 no goals
  -/


