lemma Finset.sum_indicator_mod {R : Type*} [AddCommMonoid R] (m : ℕ) [NeZero m] (f : ℕ → R) :
    f = ∑ a : ZMod m, {n : ℕ | (n : ZMod m) = a}.indicator f := by
  /-
    R : Type u_1
    inst✝¹ : AddCommMonoid R
    m : Nat
    inst✝ : NeZero m
    f : Nat → R
    ⊢ Eq f (Finset.univ.sum fun a => (setOf fun n => Eq (↑n) a).indicator f)
  -/
  ext n
  simp only [Finset.sum_apply, Set.indicator_apply, Set.mem_setOf_eq, Finset.sum_ite_eq,
    Finset.mem_univ, ↓reduceIte]


open Set in
/-- A sequence `f` with values in an additive topological group `R` is summable on the
residue class of `k` mod `m` if and only if `f (m*n + k)` is summable. -/
lemma summable_indicator_mod_iff_summable {R : Type*} [AddCommGroup R] [TopologicalSpace R]
    [TopologicalAddGroup R] (m : ℕ) [hm : NeZero m] (k : ℕ) (f : ℕ → R) :
    Summable ({n : ℕ | (n : ZMod m) = k}.indicator f) ↔ Summable fun n ↦ f (m * n + k) := by
  /-
    R : Type u_1
    inst✝² : AddCommGroup R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalAddGroup R
    m : Nat
    hm : NeZero m
    k : Nat
    f : Nat → R
    ⊢ Iff (Summable ((setOf fun n => Eq ↑n ↑k).indicator f)) (Summable fun n => f  …
  -/
  trans Summable ({n : ℕ | (n : ZMod m) = k ∧ k ≤ n}.indicator f)
    /-
      R : Type u_1
      inst✝² : AddCommGroup R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalAddGroup R
      m : Nat
      hm : NeZero m
      k : Nat
      f : Nat → R
      ⊢ Iff (Summable ((setOf fun n => Eq ↑n ↑k).indicator f)) (Summable ((setOf fun …
    -/
  · rw [← (finite_lt_nat k).summable_compl_iff (f := {n : ℕ | (n : ZMod m) = k}.indicator f)]
    simp only [summable_subtype_iff_indicator, indicator_indicator, inter_comm, setOf_and,
      compl_setOf, not_lt]
    /-
      R : Type u_1
      inst✝² : AddCommGroup R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalAddGroup R
      m : Nat
      hm : NeZero m
      k : Nat
      f : Nat → R
      ⊢ Iff (Summable ((setOf fun n => And (Eq ↑n ↑k) (LE.le k n)).indicator f)) (Su …
    -/
  · let g : ℕ → ℕ := fun n ↦ m * n + k
    /-
      R : Type u_1
      inst✝² : AddCommGroup R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalAddGroup R
      m : Nat
      hm : NeZero m
      k : Nat
      f : Nat → R
      g : Nat → Nat := fun n => HAdd.hAdd (HMul.hMul m n) k
      ⊢ Iff (Summable ((setOf fun n => And (Eq ↑n ↑k) (LE.le k n)).indicator f)) (Su …
    -/
    have hg : Function.Injective g := fun m n hmn ↦ by simpa [g, hm.ne] using hmn
    have hg' : ∀ n ∉ range g, {n : ℕ | (n : ZMod m) = k ∧ k ≤ n}.indicator f n = 0 := by
      intro n hn
      contrapose! hn
      exact (Nat.range_mul_add m k).symm ▸ mem_of_indicator_ne_zero hn
    /-
      R : Type u_1
      inst✝² : AddCommGroup R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalAddGroup R
      m : Nat
      hm : NeZero m
      k : Nat
      f : Nat → R
      g : Nat → Nat := fun n => HAdd.hAdd (HMul.hMul m n) k
      hg : Function.Injective g
      hg' : ∀ (n : Nat), Not (Membership.mem (Set.range g) n) → Eq ((setOf fun n =>  …
      ⊢ Iff (Summable ((setOf fun n => And (Eq ↑n ↑k) (LE.le k n)).indicator f)) (Su …
    -/
    convert (Function.Injective.summable_iff hg hg').symm using 3
    simp only [Function.comp_apply, mem_setOf_eq, Nat.cast_add, Nat.cast_mul, CharP.cast_eq_zero,
      zero_mul, zero_add, le_add_iff_nonneg_left, zero_le, and_self, indicator_of_mem, g]


/-- If `f : ℕ → ℝ` is decreasing and has a negative term, then `f` is not summable. -/
lemma not_summable_of_antitone_of_neg {f : ℕ → ℝ} (hf : Antitone f) {n : ℕ} (hn : f n < 0) :
    ¬ Summable f := by
  /-
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    ⊢ Not (Summable f)
  -/
  intro hs
  /-
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    ⊢ False
  -/
  have := hs.tendsto_atTop_zero
  /-
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    this : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ False
  -/
  simp only [Metric.tendsto_atTop, dist_zero_right, Real.norm_eq_abs] at this
  /-
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    this : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.l …
    ⊢ False
  -/
  obtain ⟨N, hN⟩ := this (|f n|) (abs_pos_of_neg hn)
  /-
    case intro
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    this : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.l …
    N : Nat
    hN : ∀ (n_1 : Nat), GE.ge n_1 N → LT.lt (abs (f n_1)) (abs (f n))
    ⊢ False
  -/
  specialize hN (max n N) (n.le_max_right N)
  /-
    case intro
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    this : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.l …
    N : Nat
    hN : LT.lt (abs (f (Max.max n N))) (abs (f n))
    ⊢ False
  -/
  contrapose! hN; clear hN
  /-
    case intro
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    this : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.l …
    N : Nat
    ⊢ LE.le (abs (f n)) (abs (f (Max.max n N)))
  -/
  have H : f (max n N) ≤ f n := hf (n.le_max_left N)
  /-
    case intro
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    hs : Summable f
    this : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.l …
    N : Nat
    H : LE.le (f (Max.max n N)) (f n)
    ⊢ LE.le (abs (f n)) (abs (f (Max.max n N)))
  -/
  rwa [abs_of_neg hn, abs_of_neg (H.trans_lt hn), neg_le_neg_iff]
  /-
    🎉 no goals
  -/


/-- If `f : ℕ → ℝ` is decreasing and has a negative term, then `f` restricted to a residue
class is not summable. -/
lemma not_summable_indicator_mod_of_antitone_of_neg {m : ℕ} [hm : NeZero m] {f : ℕ → ℝ}
    (hf : Antitone f) {n : ℕ} (hn : f n < 0) (k : ZMod m) :
    ¬ Summable ({n : ℕ | (n : ZMod m) = k}.indicator f) := by
  /-
    m : Nat
    hm : NeZero m
    f : Nat → Real
    hf : Antitone f
    n : Nat
    hn : LT.lt (f n) 0
    k : ZMod m
    ⊢ Not (Summable ((setOf fun n => Eq (↑n) k).indicator f))
  -/
  rw [← ZMod.natCast_zmod_val k, summable_indicator_mod_iff_summable]
  exact not_summable_of_antitone_of_neg
    (hf.comp_monotone <| (Covariant.monotone_of_const m).add_const k.val) <|
    (hf <| (Nat.le_mul_of_pos_left n Fin.pos').trans <| Nat.le_add_right ..).trans_lt hn


/-- If a decreasing sequence of real numbers is summable on one residue class
modulo `m`, then it is also summable on every other residue class mod `m`. -/
lemma summable_indicator_mod_iff_summable_indicator_mod {m : ℕ} [NeZero m] {f : ℕ → ℝ}
    (hf : Antitone f) {k : ZMod m} (l : ZMod m)
    (hs : Summable ({n : ℕ | (n : ZMod m) = k}.indicator f)) :
    Summable ({n : ℕ | (n : ZMod m) = l}.indicator f) := by
  /-
    m : Nat
    inst✝ : NeZero m
    f : Nat → Real
    hf : Antitone f
    k l : ZMod m
    hs : Summable ((setOf fun n => Eq (↑n) k).indicator f)
    ⊢ Summable ((setOf fun n => Eq (↑n) l).indicator f)
  -/
  by_cases hf₀ : ∀ n, 0 ≤ f n -- the interesting case
    /-
      case pos
      m : Nat
      inst✝ : NeZero m
      f : Nat → Real
      hf : Antitone f
      k l : ZMod m
      hs : Summable ((setOf fun n => Eq (↑n) k).indicator f)
      hf₀ : ∀ (n : Nat), LE.le 0 (f n)
      ⊢ Summable ((setOf fun n => Eq (↑n) l).indicator f)
    -/
  · rw [← ZMod.natCast_zmod_val k, summable_indicator_mod_iff_summable] at hs
    have hl : (l.val + m : ZMod m) = l := by
      simp only [ZMod.natCast_val, ZMod.cast_id', id_eq, CharP.cast_eq_zero, add_zero]
    /-
      case pos
      m : Nat
      inst✝ : NeZero m
      f : Nat → Real
      hf : Antitone f
      k l : ZMod m
      hs : Summable fun n => f (HAdd.hAdd (HMul.hMul m n) k.val)
      hf₀ : ∀ (n : Nat), LE.le 0 (f n)
      hl : Eq (HAdd.hAdd ↑l.val ↑m) l
      ⊢ Summable ((setOf fun n => Eq (↑n) l).indicator f)
    -/
    rw [← hl, ← Nat.cast_add, summable_indicator_mod_iff_summable]
    exact hs.of_nonneg_of_le (fun _ ↦ hf₀ _)
      fun _ ↦ hf <| Nat.add_le_add Nat.le.refl (k.val_lt.trans_le <| m.le_add_left l.val).le
    /-
      case neg
      m : Nat
      inst✝ : NeZero m
      f : Nat → Real
      hf : Antitone f
      k l : ZMod m
      hs : Summable ((setOf fun n => Eq (↑n) k).indicator f)
      hf₀ : Not (∀ (n : Nat), LE.le 0 (f n))
      ⊢ Summable ((setOf fun n => Eq (↑n) l).indicator f)
    -/
  · push_neg at hf₀
    /-
      case neg
      m : Nat
      inst✝ : NeZero m
      f : Nat → Real
      hf : Antitone f
      k l : ZMod m
      hs : Summable ((setOf fun n => Eq (↑n) k).indicator f)
      hf₀ : Exists fun n => LT.lt (f n) 0
      ⊢ Summable ((setOf fun n => Eq (↑n) l).indicator f)
    -/
    obtain ⟨n, hn⟩ := hf₀
    /-
      case neg.intro
      m : Nat
      inst✝ : NeZero m
      f : Nat → Real
      hf : Antitone f
      k l : ZMod m
      hs : Summable ((setOf fun n => Eq (↑n) k).indicator f)
      n : Nat
      hn : LT.lt (f n) 0
      ⊢ Summable ((setOf fun n => Eq (↑n) l).indicator f)
    -/
    exact (not_summable_indicator_mod_of_antitone_of_neg hf hn k hs).elim
    /-
      🎉 no goals
    -/


/-- A decreasing sequence of real numbers is summable on a residue class
if and only if it is summable. -/
lemma summable_indicator_mod_iff {m : ℕ} [NeZero m] {f : ℕ → ℝ} (hf : Antitone f) (k : ZMod m) :
    Summable ({n : ℕ | (n : ZMod m) = k}.indicator f) ↔ Summable f := by
  /-
    m : Nat
    inst✝ : NeZero m
    f : Nat → Real
    hf : Antitone f
    k : ZMod m
    ⊢ Iff (Summable ((setOf fun n => Eq (↑n) k).indicator f)) (Summable f)
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ Summable.indicator H _⟩
  /-
    m : Nat
    inst✝ : NeZero m
    f : Nat → Real
    hf : Antitone f
    k : ZMod m
    H : Summable ((setOf fun n => Eq (↑n) k).indicator f)
    ⊢ Summable f
  -/
  rw [Finset.sum_indicator_mod m f]
  convert summable_sum (s := Finset.univ)
    fun a _ ↦ summable_indicator_mod_iff_summable_indicator_mod hf a H
  /-
    case h.e'_5.h
    m : Nat
    inst✝ : NeZero m
    f : Nat → Real
    hf : Antitone f
    k : ZMod m
    H : Summable ((setOf fun n => Eq (↑n) k).indicator f)
    x✝ : Nat
    ⊢ Eq (Finset.univ.sum (fun a => (setOf fun n => Eq (↑n) a).indicator f) x✝) (F …
  -/
  simp only [Finset.sum_apply]
  /-
    🎉 no goals
  -/


/-- If `f` is a summable function on `ℕ`, and `0 < N`, then we may compute `∑' n : ℕ, f n` by
summing each residue class mod `N` separately. -/
lemma Nat.sumByResidueClasses {R : Type*} [AddCommGroup R] [UniformSpace R] [UniformAddGroup R]
    [CompleteSpace R] [T0Space R] {f : ℕ → R} (hf : Summable f) (N : ℕ) [NeZero N] :
    ∑' n, f n = ∑ j : ZMod N, ∑' m, f (j.val + N * m) := by
  rw [← (residueClassesEquiv N).symm.tsum_eq f, tsum_prod, tsum_fintype, residueClassesEquiv,
    Equiv.coe_fn_symm_mk]
  /-
    R : Type u_1
    inst✝⁵ : AddCommGroup R
    inst✝⁴ : UniformSpace R
    inst✝³ : UniformAddGroup R
    inst✝² : CompleteSpace R
    inst✝¹ : T0Space R
    f : Nat → R
    hf : Summable f
    N : Nat
    inst✝ : NeZero N
    ⊢ Summable fun c => f (N.residueClassesEquiv.symm c)
  -/
  exact hf.comp_injective (residueClassesEquiv N).symm.injective
  /-
    🎉 no goals
  -/

