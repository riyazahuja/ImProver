/-- `modSwap i j` contains permutations up to swapping `i` and `j`.

We use this to partition permutations in `Matrix.det_zero_of_row_eq`, such that each partition
sums up to `0`.
-/
def modSwap (i j : α) : Setoid (Perm α) :=
  ⟨fun σ τ => σ = τ ∨ σ = swap i j * τ, fun σ => Or.inl (refl σ), fun {σ τ} h =>
                                                              /-
                                                                α : Type u
                                                                inst✝ : DecidableEq α
                                                                β : Type v
                                                                i j : α
                                                                σ τ : Equiv.Perm α
                                                                h✝ : Or (Eq σ τ) (Eq σ (HMul.hMul (Equiv.swap i j) τ))
                                                                h : Eq σ (HMul.hMul (Equiv.swap i j) τ)
                                                                ⊢ Eq τ (HMul.hMul (Equiv.swap i j) σ)
                                                              -/
    Or.casesOn h (fun h => Or.inl h.symm) fun h => Or.inr (by rw [h, swap_mul_self_mul]),
                                                              /-
                                                                🎉 no goals
                                                              -/
    fun {σ τ υ} hστ hτυ => by
    /-
      α : Type u
      inst✝ : DecidableEq α
      β : Type v
      i j : α
      σ τ υ : Equiv.Perm α
      hστ : Or (Eq σ τ) (Eq σ (HMul.hMul (Equiv.swap i j) τ))
      hτυ : Or (Eq τ υ) (Eq τ (HMul.hMul (Equiv.swap i j) υ))
      ⊢ Or (Eq σ υ) (Eq σ (HMul.hMul (Equiv.swap i j) υ))
    -/
    cases' hστ with hστ hστ <;> cases' hτυ with hτυ hτυ <;> try rw [hστ, hτυ, swap_mul_self_mul] <;>
    simp [hστ, hτυ] -- Porting note: should close goals, but doesn't
      /-
        case inl.inl
        α : Type u
        inst✝ : DecidableEq α
        β : Type v
        i j : α
        σ τ υ : Equiv.Perm α
        hστ : Eq σ τ
        hτυ : Eq τ υ
        ⊢ Or (Eq σ υ) (Eq σ (HMul.hMul (Equiv.swap i j) υ))
      -/
    · simp [hστ, hτυ]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u
        inst✝ : DecidableEq α
        β : Type v
        i j : α
        σ τ υ : Equiv.Perm α
        hστ : Eq σ τ
        hτυ : Eq τ (HMul.hMul (Equiv.swap i j) υ)
        ⊢ Or (Eq σ υ) (Eq σ (HMul.hMul (Equiv.swap i j) υ))
      -/
    · simp [hστ, hτυ]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        α : Type u
        inst✝ : DecidableEq α
        β : Type v
        i j : α
        σ τ υ : Equiv.Perm α
        hστ : Eq σ (HMul.hMul (Equiv.swap i j) τ)
        hτυ : Eq τ υ
        ⊢ Or (Eq σ υ) (Eq σ (HMul.hMul (Equiv.swap i j) υ))
      -/
    · simp [hστ, hτυ]⟩
      /-
        🎉 no goals
      -/


noncomputable instance {α : Type*} [Fintype α] [DecidableEq α] (i j : α) :
    DecidableRel (modSwap i j).r :=
  fun _ _ => inferInstanceAs (Decidable (_ ∨ _))


/-- Given a list `l : List α` and a permutation `f : Perm α` such that the nonfixed points of `f`
  are in `l`, recursively factors `f` as a product of transpositions. -/
def swapFactorsAux :
    ∀ (l : List α) (f : Perm α),
      (∀ {x}, f x ≠ x → x ∈ l) → { l : List (Perm α) // l.prod = f ∧ ∀ g ∈ l, IsSwap g }
  | [] => fun f h =>
    ⟨[],
      Equiv.ext fun x => by
        /-
          α : Type u
          inst✝ : DecidableEq α
          β : Type v
          f : Equiv.Perm α
          h : ∀ {x : α}, Ne (f x) x → Membership.mem List.nil x
          x : α
          ⊢ Eq (List.nil.prod x) (f x)
        -/
        rw [List.prod_nil]
        /-
          α : Type u
          inst✝ : DecidableEq α
          β : Type v
          f : Equiv.Perm α
          h : ∀ {x : α}, Ne (f x) x → Membership.mem List.nil x
          x : α
          ⊢ Eq (1 x) (f x)
        -/
        exact (Classical.not_not.1 (mt h (List.not_mem_nil _))).symm,
        /-
          🎉 no goals
        -/
         /-
           α : Type u
           inst✝ : DecidableEq α
           β : Type v
           f : Equiv.Perm α
           h : ∀ {x : α}, Ne (f x) x → Membership.mem List.nil x
           ⊢ ∀ (g : Equiv.Perm α), Membership.mem List.nil g → g.IsSwap
         -/
      by simp⟩
         /-
           🎉 no goals
         -/
  | x::l => fun f h =>
    if hfx : x = f x then
      swapFactorsAux l f fun {y} hy =>
                                                   /-
                                                     α : Type u
                                                     inst✝ : DecidableEq α
                                                     β : Type v
                                                     x : α
                                                     l : List α
                                                     f : Equiv.Perm α
                                                     h✝ : ∀ {x_1 : α}, Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
                                                     hfx : Eq x (f x)
                                                     y : α
                                                     hy : Ne (f y) y
                                                     h : Eq y x
                                                     ⊢ False
                                                   -/
        List.mem_of_ne_of_mem (fun h : y = x => by simp [h, hfx.symm] at hy) (h hy)
                                                   /-
                                                     🎉 no goals
                                                   -/
    else
      let m :=
        swapFactorsAux l (swap x (f x) * f) fun {y} hy =>
          have : f y ≠ y ∧ y ≠ x := ne_and_ne_of_swap_mul_apply_ne_self hy
          List.mem_of_ne_of_mem this.2 (h this.1)
      ⟨swap x (f x)::m.1, by
        rw [List.prod_cons, m.2.1, ← mul_assoc, mul_def (swap x (f x)), swap_swap, ← one_def,
          one_mul],
        fun {_} hg => ((List.mem_cons).1 hg).elim (fun h => ⟨x, f x, hfx, h⟩) (m.2.2 _)⟩


/-- `swapFactors` represents a permutation as a product of a list of transpositions.
The representation is non unique and depends on the linear order structure.
For types without linear order `truncSwapFactors` can be used. -/
def swapFactors [Fintype α] [LinearOrder α] (f : Perm α) :
    { l : List (Perm α) // l.prod = f ∧ ∀ g ∈ l, IsSwap g } :=
  swapFactorsAux ((@univ α _).sort (· ≤ ·)) f fun {_ _} => (mem_sort _).2 (mem_univ _)


/-- This computably represents the fact that any permutation can be represented as the product of
  a list of transpositions. -/
def truncSwapFactors [Fintype α] (f : Perm α) :
    Trunc { l : List (Perm α) // l.prod = f ∧ ∀ g ∈ l, IsSwap g } :=
  Quotient.recOnSubsingleton (@univ α _).1 (fun l h => Trunc.mk (swapFactorsAux l f (h _)))
    (show ∀ x, f x ≠ x → x ∈ (@univ α _).1 from fun _ _ => mem_univ _)


/-- An induction principle for permutations. If `P` holds for the identity permutation, and
is preserved under composition with a non-trivial swap, then `P` holds for all permutations. -/
@[elab_as_elim]
theorem swap_induction_on [Finite α] {P : Perm α → Prop} (f : Perm α) :
    P 1 → (∀ f x y, x ≠ y → P f → P (swap x y * f)) → P f := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    P : Equiv.Perm α → Prop
    f : Equiv.Perm α
    ⊢ P 1 → (∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv.sw …
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    P : Equiv.Perm α → Prop
    f : Equiv.Perm α
    val✝ : Fintype α
    ⊢ P 1 → (∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv.sw …
  -/
  cases' (truncSwapFactors f).out with l hl
  /-
    case intro.mk
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    P : Equiv.Perm α → Prop
    f : Equiv.Perm α
    val✝ : Fintype α
    l : List (Equiv.Perm α)
    hl : And (Eq l.prod f) (∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap)
    ⊢ P 1 → (∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv.sw …
  -/
  induction' l with g l ih generalizing f
    /-
      case intro.mk.nil
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      P : Equiv.Perm α → Prop
      val✝ : Fintype α
      f : Equiv.Perm α
      hl : And (Eq List.nil.prod f) (∀ (g : Equiv.Perm α), Membership.mem List.nil g …
      ⊢ P 1 → (∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv.sw …
    -/
  · simp +contextual only [hl.left.symm, List.prod_nil, forall_true_iff]
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.cons
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      P : Equiv.Perm α → Prop
      val✝ : Fintype α
      g : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : ∀ (f : Equiv.Perm α), And (Eq l.prod f) (∀ (g : Equiv.Perm α), Membership …
      f : Equiv.Perm α
      hl : And (Eq (List.cons g l).prod f) (∀ (g_1 : Equiv.Perm α), Membership.mem ( …
      ⊢ P 1 → (∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv.sw …
    -/
  · intro h1 hmul_swap
    /-
      case intro.mk.cons
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      P : Equiv.Perm α → Prop
      val✝ : Fintype α
      g : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : ∀ (f : Equiv.Perm α), And (Eq l.prod f) (∀ (g : Equiv.Perm α), Membership …
      f : Equiv.Perm α
      hl : And (Eq (List.cons g l).prod f) (∀ (g_1 : Equiv.Perm α), Membership.mem ( …
      h1 : P 1
      hmul_swap : ∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv …
      ⊢ P f
    -/
    rcases hl.2 g (by simp) with ⟨x, y, hxy⟩
    /-
      case intro.mk.cons.intro.intro
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      P : Equiv.Perm α → Prop
      val✝ : Fintype α
      g : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : ∀ (f : Equiv.Perm α), And (Eq l.prod f) (∀ (g : Equiv.Perm α), Membership …
      f : Equiv.Perm α
      hl : And (Eq (List.cons g l).prod f) (∀ (g_1 : Equiv.Perm α), Membership.mem ( …
      h1 : P 1
      hmul_swap : ∀ (f : Equiv.Perm α) (x y : α), Ne x y → P f → P (HMul.hMul (Equiv …
      x y : α
      hxy : And (Ne x y) (Eq g (Equiv.swap x y))
      ⊢ P f
    -/
    rw [← hl.1, List.prod_cons, hxy.2]
    exact
      hmul_swap _ _ _ hxy.1
        (ih _ ⟨rfl, fun v hv => hl.2 _ (List.mem_cons_of_mem _ hv)⟩ h1 hmul_swap)


theorem mclosure_isSwap [Finite α] : Submonoid.closure { σ : Perm α | IsSwap σ } = ⊤ := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    ⊢ Eq (Submonoid.closure (setOf fun σ => σ.IsSwap)) Top.top
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    val✝ : Fintype α
    ⊢ Eq (Submonoid.closure (setOf fun σ => σ.IsSwap)) Top.top
  -/
  refine top_unique fun x _ ↦ ?_
  /-
    case intro
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    val✝ : Fintype α
    x : Equiv.Perm α
    x✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Submonoid.closure (setOf fun σ => σ.IsSwap)) x
  -/
  obtain ⟨h1, h2⟩ := Subtype.mem (truncSwapFactors x).out
  /-
    case intro.intro
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    val✝ : Fintype α
    x : Equiv.Perm α
    x✝ : Membership.mem Top.top x
    h1 : Eq (↑x.truncSwapFactors.out).prod x
    h2 : ∀ (g : Equiv.Perm α), Membership.mem (↑x.truncSwapFactors.out) g → g.IsSwap
    ⊢ Membership.mem (Submonoid.closure (setOf fun σ => σ.IsSwap)) x
  -/
  rw [← h1]
  /-
    case intro.intro
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    val✝ : Fintype α
    x : Equiv.Perm α
    x✝ : Membership.mem Top.top x
    h1 : Eq (↑x.truncSwapFactors.out).prod x
    h2 : ∀ (g : Equiv.Perm α), Membership.mem (↑x.truncSwapFactors.out) g → g.IsSwap
    ⊢ Membership.mem (Submonoid.closure (setOf fun σ => σ.IsSwap)) (↑x.truncSwapFa …
  -/
  exact Submonoid.list_prod_mem _ fun y hy ↦ Submonoid.subset_closure (h2 y hy)
  /-
    🎉 no goals
  -/


theorem closure_isSwap [Finite α] : Subgroup.closure { σ : Perm α | IsSwap σ } = ⊤ :=
  Subgroup.closure_eq_top_of_mclosure_eq_top mclosure_isSwap


/-- Every finite symmetric group is generated by transpositions of adjacent elements. -/
theorem mclosure_swap_castSucc_succ (n : ℕ) :
    Submonoid.closure (Set.range fun i : Fin n ↦ swap i.castSucc i.succ) = ⊤ := by
  /-
    n : Nat
    ⊢ Eq (Submonoid.closure (Set.range fun i => Equiv.swap i.castSucc i.succ)) Top …
  -/
  apply top_unique
  /-
    case h
    n : Nat
    ⊢ LE.le Top.top (Submonoid.closure (Set.range fun i => Equiv.swap i.castSucc i …
  -/
  rw [← mclosure_isSwap, Submonoid.closure_le]
  /-
    case h
    n : Nat
    ⊢ HasSubset.Subset (setOf fun σ => σ.IsSwap) ↑(Submonoid.closure (Set.range fu …
  -/
  rintro _ ⟨i, j, ne, rfl⟩
  /-
    case h.intro.intro.intro
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    ne : Ne i j
    ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
  -/
  wlog lt : i < j generalizing i j
    /-
      case h.intro.intro.intro.inr
      n : Nat
      i j : Fin (HAdd.hAdd n 1)
      ne : Ne i j
      this : ∀ (i j : Fin (HAdd.hAdd n 1)), Ne i j → LT.lt i j → Membership.mem (↑(S …
      lt : Not (LT.lt i j)
      ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
    -/
  · rw [swap_comm]; exact this _ _ ne.symm (ne.lt_or_lt.resolve_left lt)
                    /-
                      🎉 no goals
                    -/
  /-
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    ne : Ne i j
    lt : LT.lt i j
    ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
  -/
  induction' j using Fin.induction with j ih
    /-
      case zero
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ne : Ne i 0
      lt : LT.lt i 0
      ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
    -/
  · cases lt
    /-
      🎉 no goals
    -/
  have mem : swap j.castSucc j.succ ∈ Submonoid.closure
      (Set.range fun (i : Fin n) ↦ swap i.castSucc i.succ) := Submonoid.subset_closure ⟨_, rfl⟩
  /-
    case succ
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin n
    ih : Ne i j.castSucc → LT.lt i j.castSucc → Membership.mem (↑(Submonoid.closur …
    ne : Ne i j.succ
    lt : LT.lt i j.succ
    mem : Membership.mem (Submonoid.closure (Set.range fun i => Equiv.swap i.castS …
    ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
  -/
  obtain rfl | lts := (Fin.le_castSucc_iff.mpr lt).eq_or_lt
    /-
      case succ.inl
      n : Nat
      j : Fin n
      mem : Membership.mem (Submonoid.closure (Set.range fun i => Equiv.swap i.castS …
      ih : Ne j.castSucc j.castSucc → LT.lt j.castSucc j.castSucc → Membership.mem ( …
      ne : Ne j.castSucc j.succ
      lt : LT.lt j.castSucc j.succ
      ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
    -/
  · exact mem
    /-
      🎉 no goals
    -/
  /-
    case succ.inr
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin n
    ih : Ne i j.castSucc → LT.lt i j.castSucc → Membership.mem (↑(Submonoid.closur …
    ne : Ne i j.succ
    lt : LT.lt i j.succ
    mem : Membership.mem (Submonoid.closure (Set.range fun i => Equiv.swap i.castS …
    lts : LT.lt i j.castSucc
    ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
  -/
  rw [swap_comm, ← swap_mul_swap_mul_swap (y := Fin.castSucc j) lts.ne lt.ne]
  /-
    case succ.inr
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin n
    ih : Ne i j.castSucc → LT.lt i j.castSucc → Membership.mem (↑(Submonoid.closur …
    ne : Ne i j.succ
    lt : LT.lt i j.succ
    mem : Membership.mem (Submonoid.closure (Set.range fun i => Equiv.swap i.castS …
    lts : LT.lt i j.castSucc
    ⊢ Membership.mem (↑(Submonoid.closure (Set.range fun i => Equiv.swap i.castSuc …
  -/
  exact mul_mem (mul_mem mem <| ih lts.ne lts) mem
  /-
    🎉 no goals
  -/


/-- Like `swap_induction_on`, but with the composition on the right of `f`.

An induction principle for permutations. If `P` holds for the identity permutation, and
is preserved under composition with a non-trivial swap, then `P` holds for all permutations. -/
@[elab_as_elim]
theorem swap_induction_on' [Finite α] {P : Perm α → Prop} (f : Perm α) :
    P 1 → (∀ f x y, x ≠ y → P f → P (f * swap x y)) → P f := fun h1 IH =>
  inv_inv f ▸ swap_induction_on f⁻¹ h1 fun f => IH f⁻¹


theorem isConj_swap {w x y z : α} (hwx : w ≠ x) (hyz : y ≠ z) : IsConj (swap w x) (swap y z) :=
  isConj_iff.2
    (have h :
      ∀ {y z : α},
        y ≠ z → w ≠ z → swap w y * swap x z * swap w x * (swap w y * swap x z)⁻¹ = swap y z :=
      fun {y z} hyz hwz => by
      rw [mul_inv_rev, swap_inv, swap_inv, mul_assoc (swap w y), mul_assoc (swap w y), ←
        mul_assoc _ (swap x z), swap_mul_swap_mul_swap hwx hwz, ← mul_assoc,
        swap_mul_swap_mul_swap hwz.symm hyz.symm]
    if hwz : w = z then
                             /-
                               α : Type u
                               inst✝ : DecidableEq α
                               w x y z : α
                               hwx : Ne w x
                               hyz : Ne y z
                               h : ∀ {y z : α}, Ne y z → Ne w z → Eq (HMul.hMul (HMul.hMul (HMul.hMul (Equiv. …
                               hwz : Eq w z
                               ⊢ Ne w y
                             -/
      have hwy : w ≠ y := by rw [hwz]; exact hyz.symm
                                       /-
                                         🎉 no goals
                                       -/
                               /-
                                 α : Type u
                                 inst✝ : DecidableEq α
                                 w x y z : α
                                 hwx : Ne w x
                                 hyz : Ne y z
                                 h : ∀ {y z : α}, Ne y z → Ne w z → Eq (HMul.hMul (HMul.hMul (HMul.hMul (Equiv. …
                                 hwz : Eq w z
                                 hwy : Ne w y
                                 ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Equiv.swap w z) (Equiv.swap x y)) (Equi …
                               -/
      ⟨swap w z * swap x y, by rw [swap_comm y z, h hyz.symm hwy]⟩
                               /-
                                 🎉 no goals
                               -/
    else ⟨swap w y * swap x z, h hyz hwz⟩)


/-- set of all pairs (⟨a, b⟩ : Σ a : fin n, fin n) such that b < a -/
def finPairsLT (n : ℕ) : Finset (Σ_ : Fin n, Fin n) :=
  (univ : Finset (Fin n)).sigma fun a => (range a).attachFin fun _ hm => (mem_range.1 hm).trans a.2


theorem mem_finPairsLT {n : ℕ} {a : Σ_ : Fin n, Fin n} : a ∈ finPairsLT n ↔ a.2 < a.1 := by
  simp only [finPairsLT, Fin.lt_iff_val_lt_val, true_and, mem_attachFin, mem_range, mem_univ,
    mem_sigma]


/-- `signAux σ` is the sign of a permutation on `Fin n`, defined as the parity of the number of
  pairs `(x₁, x₂)` such that `x₂ < x₁` but `σ x₁ ≤ σ x₂` -/
def signAux {n : ℕ} (a : Perm (Fin n)) : ℤˣ :=
  ∏ x ∈ finPairsLT n, if a x.1 ≤ a x.2 then -1 else 1


@[simp]
theorem signAux_one (n : ℕ) : signAux (1 : Perm (Fin n)) = 1 := by
  /-
    n : Nat
    ⊢ Eq (Equiv.Perm.signAux 1) 1
  -/
  unfold signAux
  /-
    n : Nat
    ⊢ Eq ((Equiv.Perm.finPairsLT n).prod fun x => ite (LE.le (1 x.fst) (1 x.snd))  …
  -/
  conv => rhs; rw [← @Finset.prod_const_one _ _ (finPairsLT n)]
  /-
    n : Nat
    ⊢ Eq ((Equiv.Perm.finPairsLT n).prod fun x => ite (LE.le (1 x.fst) (1 x.snd))  …
  -/
  exact Finset.prod_congr rfl fun a ha => if_neg (mem_finPairsLT.1 ha).not_le
  /-
    🎉 no goals
  -/


/-- `signBijAux f ⟨a, b⟩` returns the pair consisting of `f a` and `f b` in decreasing order. -/
def signBijAux {n : ℕ} (f : Perm (Fin n)) (a : Σ_ : Fin n, Fin n) : Σ_ : Fin n, Fin n :=
  if _ : f a.2 < f a.1 then ⟨f a.1, f a.2⟩ else ⟨f a.2, f a.1⟩


theorem signBijAux_injOn {n : ℕ} {f : Perm (Fin n)} :
    (finPairsLT n : Set (Σ _, Fin n)).InjOn (signBijAux f) := by
  /-
    n : Nat
    f : Equiv.Perm (Fin n)
    ⊢ Set.InjOn f.signBijAux ↑(Equiv.Perm.finPairsLT n)
  -/
  rintro ⟨a₁, a₂⟩ ha ⟨b₁, b₂⟩ hb h
  /-
    case mk.mk
    n : Nat
    f : Equiv.Perm (Fin n)
    a₁ a₂ : Fin n
    ha : Membership.mem ↑(Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
    b₁ b₂ : Fin n
    hb : Membership.mem ↑(Equiv.Perm.finPairsLT n) ⟨b₁, b₂⟩
    h : Eq (f.signBijAux ⟨a₁, a₂⟩) (f.signBijAux ⟨b₁, b₂⟩)
    ⊢ Eq ⟨a₁, a₂⟩ ⟨b₁, b₂⟩
  -/
  dsimp [signBijAux] at h
  /-
    case mk.mk
    n : Nat
    f : Equiv.Perm (Fin n)
    a₁ a₂ : Fin n
    ha : Membership.mem ↑(Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
    b₁ b₂ : Fin n
    hb : Membership.mem ↑(Equiv.Perm.finPairsLT n) ⟨b₁, b₂⟩
    h : Eq (ite (LT.lt (f a₂) (f a₁)) ⟨f a₁, f a₂⟩ ⟨f a₂, f a₁⟩) (ite (LT.lt (f b₂ …
    ⊢ Eq ⟨a₁, a₂⟩ ⟨b₁, b₂⟩
  -/
  rw [Finset.mem_coe, mem_finPairsLT] at *
  /-
    case mk.mk
    n : Nat
    f : Equiv.Perm (Fin n)
    a₁ a₂ : Fin n
    ha : LT.lt ⟨a₁, a₂⟩.snd ⟨a₁, a₂⟩.fst
    b₁ b₂ : Fin n
    hb : LT.lt ⟨b₁, b₂⟩.snd ⟨b₁, b₂⟩.fst
    h : Eq (ite (LT.lt (f a₂) (f a₁)) ⟨f a₁, f a₂⟩ ⟨f a₂, f a₁⟩) (ite (LT.lt (f b₂ …
    ⊢ Eq ⟨a₁, a₂⟩ ⟨b₁, b₂⟩
  -/
  have : ¬b₁ < b₂ := hb.le.not_lt
  /-
    case mk.mk
    n : Nat
    f : Equiv.Perm (Fin n)
    a₁ a₂ : Fin n
    ha : LT.lt ⟨a₁, a₂⟩.snd ⟨a₁, a₂⟩.fst
    b₁ b₂ : Fin n
    hb : LT.lt ⟨b₁, b₂⟩.snd ⟨b₁, b₂⟩.fst
    h : Eq (ite (LT.lt (f a₂) (f a₁)) ⟨f a₁, f a₂⟩ ⟨f a₂, f a₁⟩) (ite (LT.lt (f b₂ …
    this : Not (LT.lt b₁ b₂)
    ⊢ Eq ⟨a₁, a₂⟩ ⟨b₁, b₂⟩
  -/
  split_ifs at h <;>
  /-
    case pos
    n : Nat
    f : Equiv.Perm (Fin n)
    a₁ a₂ : Fin n
    ha : LT.lt ⟨a₁, a₂⟩.snd ⟨a₁, a₂⟩.fst
    b₁ b₂ : Fin n
    hb : LT.lt ⟨b₁, b₂⟩.snd ⟨b₁, b₂⟩.fst
    this : Not (LT.lt b₁ b₂)
    h✝¹ : LT.lt (f a₂) (f a₁)
    h✝ : LT.lt (f b₂) (f b₁)
    h : Eq ⟨f a₁, f a₂⟩ ⟨f b₁, f b₂⟩
    ⊢ Eq ⟨a₁, a₂⟩ ⟨b₁, b₂⟩
  -/
  /-
    🎉 no goals
  -/
  simp_all only [not_lt, Sigma.mk.inj_iff, (Equiv.injective f).eq_iff, heq_eq_eq]
  /-
    🎉 no goals
  -/
    /-
      case neg
      n : Nat
      f : Equiv.Perm (Fin n)
      a₁ a₂ b₁ b₂ : Fin n
      ha : LT.lt b₁ b₂
      hb : LT.lt b₂ b₁
      this : LE.le b₂ b₁
      h✝¹ : LT.lt (f b₁) (f b₂)
      h✝ : LE.le (f b₁) (f b₂)
      h : And (Eq a₁ b₂) (Eq a₂ b₁)
      ⊢ And (Eq b₂ b₁) (Eq b₁ b₂)
    -/
  · exact absurd this (not_le.mpr ha)
    /-
      🎉 no goals
    -/
    /-
      case pos
      n : Nat
      f : Equiv.Perm (Fin n)
      a₁ a₂ b₁ b₂ : Fin n
      ha : LT.lt b₁ b₂
      hb : LT.lt b₂ b₁
      this : LE.le b₂ b₁
      h✝¹ : LE.le (f b₂) (f b₁)
      h✝ : LT.lt (f b₂) (f b₁)
      h : And (Eq a₂ b₁) (Eq a₁ b₂)
      ⊢ And (Eq b₂ b₁) (Eq b₁ b₂)
    -/
  · exact absurd this (not_le.mpr ha)
    /-
      🎉 no goals
    -/


theorem signBijAux_surj {n : ℕ} {f : Perm (Fin n)} :
    ∀ a ∈ finPairsLT n, ∃ b ∈ finPairsLT n, signBijAux f b = a :=
  fun ⟨a₁, a₂⟩ ha =>
    if hxa : f⁻¹ a₂ < f⁻¹ a₁ then
      ⟨⟨f⁻¹ a₁, f⁻¹ a₂⟩, mem_finPairsLT.2 hxa, by
        /-
          n : Nat
          f : Equiv.Perm (Fin n)
          x✝ : Sigma fun x => Fin n
          a₁ a₂ : Fin n
          ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
          hxa : LT.lt ((Inv.inv f) a₂) ((Inv.inv f) a₁)
          ⊢ Eq (f.signBijAux ⟨(Inv.inv f) a₁, (Inv.inv f) a₂⟩) ⟨a₁, a₂⟩
        -/
        dsimp [signBijAux]
        /-
          n : Nat
          f : Equiv.Perm (Fin n)
          x✝ : Sigma fun x => Fin n
          a₁ a₂ : Fin n
          ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
          hxa : LT.lt ((Inv.inv f) a₂) ((Inv.inv f) a₁)
          ⊢ Eq (ite (LT.lt (f ((Inv.inv f) a₂)) (f ((Inv.inv f) a₁))) ⟨f ((Inv.inv f) a₁ …
        -/
        rw [apply_inv_self, apply_inv_self, if_pos (mem_finPairsLT.1 ha)]⟩
        /-
          🎉 no goals
        -/
    else
      ⟨⟨f⁻¹ a₂, f⁻¹ a₁⟩,
        mem_finPairsLT.2 <|
          (le_of_not_gt hxa).lt_of_ne fun h => by
            /-
              n : Nat
              f : Equiv.Perm (Fin n)
              x✝ : Sigma fun x => Fin n
              a₁ a₂ : Fin n
              ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
              hxa : Not (LT.lt ((Inv.inv f) a₂) ((Inv.inv f) a₁))
              h : Eq ((Inv.inv f) a₁) ((Inv.inv f) a₂)
              ⊢ False
            -/
            simp [mem_finPairsLT, f⁻¹.injective h, lt_irrefl] at ha, by
            /-
              🎉 no goals
            -/
              /-
                n : Nat
                f : Equiv.Perm (Fin n)
                x✝ : Sigma fun x => Fin n
                a₁ a₂ : Fin n
                ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
                hxa : Not (LT.lt ((Inv.inv f) a₂) ((Inv.inv f) a₁))
                ⊢ Eq (f.signBijAux ⟨(Inv.inv f) a₂, (Inv.inv f) a₁⟩) ⟨a₁, a₂⟩
              -/
              dsimp [signBijAux]
              /-
                n : Nat
                f : Equiv.Perm (Fin n)
                x✝ : Sigma fun x => Fin n
                a₁ a₂ : Fin n
                ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
                hxa : Not (LT.lt ((Inv.inv f) a₂) ((Inv.inv f) a₁))
                ⊢ Eq (ite (LT.lt (f ((Inv.inv f) a₁)) (f ((Inv.inv f) a₂))) ⟨f ((Inv.inv f) a₂ …
              -/
              rw [apply_inv_self, apply_inv_self, if_neg (mem_finPairsLT.1 ha).le.not_lt]⟩
              /-
                🎉 no goals
              -/


theorem signBijAux_mem {n : ℕ} {f : Perm (Fin n)} :
    ∀ a : Σ_ : Fin n, Fin n, a ∈ finPairsLT n → signBijAux f a ∈ finPairsLT n :=
  fun ⟨a₁, a₂⟩ ha => by
    /-
      n : Nat
      f : Equiv.Perm (Fin n)
      x✝ : Sigma fun x => Fin n
      a₁ a₂ : Fin n
      ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
      ⊢ Membership.mem (Equiv.Perm.finPairsLT n) (f.signBijAux ⟨a₁, a₂⟩)
    -/
    unfold signBijAux
    /-
      n : Nat
      f : Equiv.Perm (Fin n)
      x✝ : Sigma fun x => Fin n
      a₁ a₂ : Fin n
      ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
      ⊢ Membership.mem (Equiv.Perm.finPairsLT n) (dite (LT.lt (f ⟨a₁, a₂⟩.snd) (f ⟨a …
    -/
    split_ifs with h
      /-
        case pos
        n : Nat
        f : Equiv.Perm (Fin n)
        x✝ : Sigma fun x => Fin n
        a₁ a₂ : Fin n
        ha : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a₁, a₂⟩
        h : LT.lt (f ⟨a₁, a₂⟩.snd) (f ⟨a₁, a₂⟩.fst)
        ⊢ Membership.mem (Equiv.Perm.finPairsLT n) ⟨f ⟨a₁, a₂⟩.fst, f ⟨a₁, a₂⟩.snd⟩
      -/
    · exact mem_finPairsLT.2 h
      /-
        🎉 no goals
      -/
    · exact mem_finPairsLT.2
        ((le_of_not_gt h).lt_of_ne fun h => (mem_finPairsLT.1 ha).ne (f.injective h.symm))


@[simp]
theorem signAux_inv {n : ℕ} (f : Perm (Fin n)) : signAux f⁻¹ = signAux f :=
  prod_nbij (signBijAux f⁻¹) signBijAux_mem signBijAux_injOn signBijAux_surj fun ⟨a, b⟩ hab ↦
    if h : f⁻¹ b < f⁻¹ a then by
      simp_all [signBijAux, dif_pos h, if_neg h.not_le, apply_inv_self, apply_inv_self,
        if_neg (mem_finPairsLT.1 hab).not_le]
    else by
      simp_all [signBijAux, if_pos (le_of_not_gt h), dif_neg h, apply_inv_self, apply_inv_self,
        if_pos (mem_finPairsLT.1 hab).le]


theorem signAux_mul {n : ℕ} (f g : Perm (Fin n)) : signAux (f * g) = signAux f * signAux g := by
  /-
    n : Nat
    f g : Equiv.Perm (Fin n)
    ⊢ Eq (HMul.hMul f g).signAux (HMul.hMul f.signAux g.signAux)
  -/
  rw [← signAux_inv g]
  /-
    n : Nat
    f g : Equiv.Perm (Fin n)
    ⊢ Eq (HMul.hMul f g).signAux (HMul.hMul f.signAux (Inv.inv g).signAux)
  -/
  unfold signAux
  /-
    n : Nat
    f g : Equiv.Perm (Fin n)
    ⊢ Eq ((Equiv.Perm.finPairsLT n).prod fun x => ite (LE.le ((HMul.hMul f g) x.fs …
  -/
  rw [← prod_mul_distrib]
  /-
    n : Nat
    f g : Equiv.Perm (Fin n)
    ⊢ Eq ((Equiv.Perm.finPairsLT n).prod fun x => ite (LE.le ((HMul.hMul f g) x.fs …
  -/
  refine prod_nbij (signBijAux g) signBijAux_mem signBijAux_injOn signBijAux_surj ?_
  /-
    n : Nat
    f g : Equiv.Perm (Fin n)
    ⊢ ∀ (a : Sigma fun x => Fin n), Membership.mem (Equiv.Perm.finPairsLT n) a → E …
  -/
  rintro ⟨a, b⟩ hab
  /-
    case mk
    n : Nat
    f g : Equiv.Perm (Fin n)
    a b : Fin n
    hab : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a, b⟩
    ⊢ Eq (ite (LE.le ((HMul.hMul f g) ⟨a, b⟩.fst) ((HMul.hMul f g) ⟨a, b⟩.snd)) (- …
  -/
  dsimp only [signBijAux]
  /-
    case mk
    n : Nat
    f g : Equiv.Perm (Fin n)
    a b : Fin n
    hab : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a, b⟩
    ⊢ Eq (ite (LE.le ((HMul.hMul f g) a) ((HMul.hMul f g) b)) (-1) 1) (HMul.hMul ( …
  -/
  rw [mul_apply, mul_apply]
  /-
    case mk
    n : Nat
    f g : Equiv.Perm (Fin n)
    a b : Fin n
    hab : Membership.mem (Equiv.Perm.finPairsLT n) ⟨a, b⟩
    ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f (dite  …
  -/
  rw [mem_finPairsLT] at hab
  /-
    case mk
    n : Nat
    f g : Equiv.Perm (Fin n)
    a b : Fin n
    hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
    ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f (dite  …
  -/
  by_cases h : g b < g a
    /-
      case pos
      n : Nat
      f g : Equiv.Perm (Fin n)
      a b : Fin n
      hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
      h : LT.lt (g b) (g a)
      ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f (dite  …
    -/
  · rw [dif_pos h]
    /-
      case pos
      n : Nat
      f g : Equiv.Perm (Fin n)
      a b : Fin n
      hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
      h : LT.lt (g b) (g a)
      ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f ⟨g a,  …
    -/
    simp only [not_le_of_gt hab, mul_one, mul_ite, mul_neg, Perm.inv_apply_self, if_false]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      f g : Equiv.Perm (Fin n)
      a b : Fin n
      hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
      h : Not (LT.lt (g b) (g a))
      ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f (dite  …
    -/
  · rw [dif_neg h, inv_apply_self, inv_apply_self, if_pos hab.le]
    /-
      case neg
      n : Nat
      f g : Equiv.Perm (Fin n)
      a b : Fin n
      hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
      h : Not (LT.lt (g b) (g a))
      ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f ⟨g b,  …
    -/
    by_cases h₁ : f (g b) ≤ f (g a)
    · have : f (g b) ≠ f (g a) := by
        rw [Ne, f.injective.eq_iff, g.injective.eq_iff]
        exact ne_of_lt hab
      /-
        case pos
        n : Nat
        f g : Equiv.Perm (Fin n)
        a b : Fin n
        hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
        h : Not (LT.lt (g b) (g a))
        h₁ : LE.le (f (g b)) (f (g a))
        this : Ne (f (g b)) (f (g a))
        ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f ⟨g b,  …
      -/
      rw [if_pos h₁, if_neg (h₁.lt_of_ne this).not_le]
      /-
        case pos
        n : Nat
        f g : Equiv.Perm (Fin n)
        a b : Fin n
        hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
        h : Not (LT.lt (g b) (g a))
        h₁ : LE.le (f (g b)) (f (g a))
        this : Ne (f (g b)) (f (g a))
        ⊢ Eq 1 (HMul.hMul (-1) (-1))
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        n : Nat
        f g : Equiv.Perm (Fin n)
        a b : Fin n
        hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
        h : Not (LT.lt (g b) (g a))
        h₁ : Not (LE.le (f (g b)) (f (g a)))
        ⊢ Eq (ite (LE.le (f (g a)) (f (g b))) (-1) 1) (HMul.hMul (ite (LE.le (f ⟨g b,  …
      -/
    · rw [if_neg h₁, if_pos (lt_of_not_ge h₁).le]
      /-
        case neg
        n : Nat
        f g : Equiv.Perm (Fin n)
        a b : Fin n
        hab : LT.lt ⟨a, b⟩.snd ⟨a, b⟩.fst
        h : Not (LT.lt (g b) (g a))
        h₁ : Not (LE.le (f (g b)) (f (g a)))
        ⊢ Eq (-1) (HMul.hMul 1 (-1))
      -/
      rfl
      /-
        🎉 no goals
      -/


private theorem signAux_swap_zero_one' (n : ℕ) : signAux (swap (0 : Fin (n + 2)) 1) = -1 :=
  show _ = ∏ x ∈ {(⟨1, 0⟩ : Σ _ : Fin (n + 2), Fin (n + 2))},
      if (Equiv.swap 0 1) x.1 ≤ swap 0 1 x.2 then (-1 : ℤˣ) else 1 by
    refine Eq.symm (prod_subset (fun ⟨x₁, x₂⟩ => by
      simp +contextual [mem_finPairsLT, Fin.one_pos]) fun a ha₁ ha₂ => ?_)
    /-
      n : Nat
      a : Sigma fun x => Fin (HAdd.hAdd n 2)
      ha₁ : Membership.mem (Equiv.Perm.finPairsLT (HAdd.hAdd n 2)) a
      ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) a)
      ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a.fst) ((Equiv.swap 0 1) a.snd)) (-1) 1) 1
    -/
    rcases a with ⟨a₁, a₂⟩
    /-
      case mk
      n : Nat
      a₁ a₂ : Fin (HAdd.hAdd n 2)
      ha₁ : Membership.mem (Equiv.Perm.finPairsLT (HAdd.hAdd n 2)) ⟨a₁, a₂⟩
      ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
      ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) ⟨a₁, a₂⟩.fst) ((Equiv.swap 0 1) ⟨a₁, a₂⟩.sn …
    -/
    replace ha₁ : a₂ < a₁ := mem_finPairsLT.1 ha₁
    /-
      case mk
      n : Nat
      a₁ a₂ : Fin (HAdd.hAdd n 2)
      ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
      ha₁ : LT.lt a₂ a₁
      ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) ⟨a₁, a₂⟩.fst) ((Equiv.swap 0 1) ⟨a₁, a₂⟩.sn …
    -/
    dsimp only
    /-
      case mk
      n : Nat
      a₁ a₂ : Fin (HAdd.hAdd n 2)
      ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
      ha₁ : LT.lt a₂ a₁
      ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
    -/
    rcases a₁.zero_le.eq_or_lt with (rfl | H)
      /-
        case mk.inl
        n : Nat
        a₂ : Fin (HAdd.hAdd n 2)
        ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨0, a₂⟩)
        ha₁ : LT.lt a₂ 0
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) 0) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
      -/
    · exact absurd a₂.zero_le ha₁.not_le
      /-
        🎉 no goals
      -/
    /-
      case mk.inr
      n : Nat
      a₁ a₂ : Fin (HAdd.hAdd n 2)
      ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
      ha₁ : LT.lt a₂ a₁
      H : LT.lt 0 a₁
      ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
    -/
    rcases a₂.zero_le.eq_or_lt with (rfl | H')
      /-
        case mk.inr.inl
        n : Nat
        a₁ : Fin (HAdd.hAdd n 2)
        H : LT.lt 0 a₁
        ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, 0⟩)
        ha₁ : LT.lt 0 a₁
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) 0)) (-1) 1) 1
      -/
    · simp only [and_true, eq_self_iff_true, heq_iff_eq, mem_singleton, Sigma.mk.inj_iff] at ha₂
      have : 1 < a₁ := lt_of_le_of_ne (Nat.succ_le_of_lt ha₁)
        (Ne.symm (by intro h; apply ha₂; simp [h]))
      /-
        case mk.inr.inl
        n : Nat
        a₁ : Fin (HAdd.hAdd n 2)
        H : LT.lt 0 a₁
        ha₁ : LT.lt 0 a₁
        ha₂ : Not (Eq a₁ 1)
        this : LT.lt 1 a₁
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) 0)) (-1) 1) 1
      -/
      have h01 : Equiv.swap (0 : Fin (n + 2)) 1 0 = 1 := by simp
      /-
        case mk.inr.inl
        n : Nat
        a₁ : Fin (HAdd.hAdd n 2)
        H : LT.lt 0 a₁
        ha₁ : LT.lt 0 a₁
        ha₂ : Not (Eq a₁ 1)
        this : LT.lt 1 a₁
        h01 : Eq ((Equiv.swap 0 1) 0) 1
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) 0)) (-1) 1) 1
      -/
      rw [swap_apply_of_ne_of_ne (ne_of_gt H) ha₂, h01, if_neg this.not_le]
      /-
        🎉 no goals
      -/
      /-
        case mk.inr.inr
        n : Nat
        a₁ a₂ : Fin (HAdd.hAdd n 2)
        ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
        ha₁ : LT.lt a₂ a₁
        H : LT.lt 0 a₁
        H' : LT.lt 0 a₂
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
      -/
    · have le : 1 ≤ a₂ := Nat.succ_le_of_lt H'
      /-
        case mk.inr.inr
        n : Nat
        a₁ a₂ : Fin (HAdd.hAdd n 2)
        ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
        ha₁ : LT.lt a₂ a₁
        H : LT.lt 0 a₁
        H' : LT.lt 0 a₂
        le : LE.le 1 a₂
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
      -/
      have lt : 1 < a₁ := le.trans_lt ha₁
      /-
        case mk.inr.inr
        n : Nat
        a₁ a₂ : Fin (HAdd.hAdd n 2)
        ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
        ha₁ : LT.lt a₂ a₁
        H : LT.lt 0 a₁
        H' : LT.lt 0 a₂
        le : LE.le 1 a₂
        lt : LT.lt 1 a₁
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
      -/
      have h01 : Equiv.swap (0 : Fin (n + 2)) 1 1 = 0 := by simp only [swap_apply_right]
      /-
        case mk.inr.inr
        n : Nat
        a₁ a₂ : Fin (HAdd.hAdd n 2)
        ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, a₂⟩)
        ha₁ : LT.lt a₂ a₁
        H : LT.lt 0 a₁
        H' : LT.lt 0 a₂
        le : LE.le 1 a₂
        lt : LT.lt 1 a₁
        h01 : Eq ((Equiv.swap 0 1) 1) 0
        ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) a₂)) (-1) 1) 1
      -/
      rcases le.eq_or_lt with (rfl | lt')
        /-
          case mk.inr.inr.inl
          n : Nat
          a₁ : Fin (HAdd.hAdd n 2)
          H : LT.lt 0 a₁
          lt : LT.lt 1 a₁
          h01 : Eq ((Equiv.swap 0 1) 1) 0
          ha₂ : Not (Membership.mem (Singleton.singleton ⟨1, 0⟩) ⟨a₁, 1⟩)
          ha₁ : LT.lt 1 a₁
          H' : LT.lt 0 1
          le : LE.le 1 1
          ⊢ Eq (ite (LE.le ((Equiv.swap 0 1) a₁) ((Equiv.swap 0 1) 1)) (-1) 1) 1
        -/
      · rw [swap_apply_of_ne_of_ne H.ne' lt.ne', h01, if_neg H.not_le]
        /-
          🎉 no goals
        -/
      · rw [swap_apply_of_ne_of_ne (ne_of_gt H) (ne_of_gt lt),
          swap_apply_of_ne_of_ne (ne_of_gt H') (ne_of_gt lt'), if_neg ha₁.not_le]


private theorem signAux_swap_zero_one {n : ℕ} (hn : 2 ≤ n) :
                                          /-
                                            α : Type u
                                            inst✝ : DecidableEq α
                                            β : Type v
                                            n : Nat
                                            hn : LE.le 2 n
                                            ⊢ LT.lt 0 2
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
    signAux (swap (⟨0, lt_of_lt_of_le (by decide) hn⟩ : Fin n) ⟨1, lt_of_lt_of_le (by decide) hn⟩) =
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
      -1 := by
  /-
    n : Nat
    hn : LE.le 2 n
    ⊢ Eq (Equiv.swap ⟨0, ⋯⟩ ⟨1, ⋯⟩).signAux (-1)
  -/
  rcases n with (_ | _ | n)
    /-
      case zero
      hn : LE.le 2 0
      ⊢ Eq (Equiv.swap ⟨0, ⋯⟩ ⟨1, ⋯⟩).signAux (-1)
    -/
  · norm_num at hn
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      hn : LE.le 2 (HAdd.hAdd 0 1)
      ⊢ Eq (Equiv.swap ⟨0, ⋯⟩ ⟨1, ⋯⟩).signAux (-1)
    -/
  · norm_num at hn
    /-
      🎉 no goals
    -/
    /-
      case succ.succ
      n : Nat
      hn : LE.le 2 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (Equiv.swap ⟨0, ⋯⟩ ⟨1, ⋯⟩).signAux (-1)
    -/
  · exact signAux_swap_zero_one' n
    /-
      🎉 no goals
    -/


theorem signAux_swap : ∀ {n : ℕ} {x y : Fin n} (_hxy : x ≠ y), signAux (swap x y) = -1
                  /-
                    x y : Fin 0
                    ⊢ Ne x y → Eq (Equiv.swap x y).signAux (-1)
                  -/
  | 0, x, y => by intro; exact Fin.elim0 x
                         /-
                           🎉 no goals
                         -/
  | 1, x, y => by
    /-
      x y : Fin 1
      ⊢ Ne x y → Eq (Equiv.swap x y).signAux (-1)
    -/
    dsimp [signAux, swap, swapCore]
    simp only [eq_iff_true_of_subsingleton, not_true, ite_true, le_refl, prod_const,
               IsEmpty.forall_iff]
  | n + 2, x, y => fun hxy => by
    /-
      n : Nat
      x y : Fin (HAdd.hAdd n 2)
      hxy : Ne x y
      ⊢ Eq (Equiv.swap x y).signAux (-1)
    -/
    have h2n : 2 ≤ n + 2 := by exact le_add_self
    /-
      n : Nat
      x y : Fin (HAdd.hAdd n 2)
      hxy : Ne x y
      h2n : LE.le 2 (HAdd.hAdd n 2)
      ⊢ Eq (Equiv.swap x y).signAux (-1)
    -/
    rw [← isConj_iff_eq, ← signAux_swap_zero_one h2n]
    exact (MonoidHom.mk' signAux signAux_mul).map_isConj
      (isConj_swap hxy (by exact of_decide_eq_true rfl))


/-- When the list `l : List α` contains all nonfixed points of the permutation `f : Perm α`,
  `signAux2 l f` recursively calculates the sign of `f`. -/
def signAux2 : List α → Perm α → ℤˣ
  | [], _ => 1
  | x::l, f => if x = f x then signAux2 l f else -signAux2 l (swap x (f x) * f)


theorem signAux_eq_signAux2 {n : ℕ} :
    ∀ (l : List α) (f : Perm α) (e : α ≃ Fin n) (_h : ∀ x, f x ≠ x → x ∈ l),
      signAux ((e.symm.trans f).trans e) = signAux2 l f
  | [], f, e, h => by
    /-
      α : Type u
      inst✝ : DecidableEq α
      n : Nat
      f : Equiv.Perm α
      e : Equiv α (Fin n)
      h : ∀ (x : α), Ne (f x) x → Membership.mem List.nil x
      ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (Equiv.Perm.signAux2 List …
    -/
    have : f = 1 := Equiv.ext fun y => Classical.not_not.1 (mt (h y) (List.not_mem_nil _))
    /-
      α : Type u
      inst✝ : DecidableEq α
      n : Nat
      f : Equiv.Perm α
      e : Equiv α (Fin n)
      h : ∀ (x : α), Ne (f x) x → Membership.mem List.nil x
      this : Eq f 1
      ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (Equiv.Perm.signAux2 List …
    -/
    rw [this, one_def, Equiv.trans_refl, Equiv.symm_trans_self, ← one_def, signAux_one, signAux2]
    /-
      🎉 no goals
    -/
  | x::l, f, e, h => by
    /-
      α : Type u
      inst✝ : DecidableEq α
      n : Nat
      x : α
      l : List α
      f : Equiv.Perm α
      e : Equiv α (Fin n)
      h : ∀ (x_1 : α), Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
      ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (Equiv.Perm.signAux2 (Lis …
    -/
    rw [signAux2]
    /-
      α : Type u
      inst✝ : DecidableEq α
      n : Nat
      x : α
      l : List α
      f : Equiv.Perm α
      e : Equiv α (Fin n)
      h : ∀ (x_1 : α), Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
      ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (ite (Eq x (f x)) (Equiv. …
    -/
    by_cases hfx : x = f x
      /-
        case pos
        α : Type u
        inst✝ : DecidableEq α
        n : Nat
        x : α
        l : List α
        f : Equiv.Perm α
        e : Equiv α (Fin n)
        h : ∀ (x_1 : α), Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
        hfx : Eq x (f x)
        ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (ite (Eq x (f x)) (Equiv. …
      -/
    · rw [if_pos hfx]
      exact
        signAux_eq_signAux2 l f _ fun y (hy : f y ≠ y) =>
          List.mem_of_ne_of_mem (fun h : y = x => by simp [h, hfx.symm] at hy) (h y hy)
    · have hy : ∀ y : α, (swap x (f x) * f) y ≠ y → y ∈ l := fun y hy =>
        have : f y ≠ y ∧ y ≠ x := ne_and_ne_of_swap_mul_apply_ne_self hy
        List.mem_of_ne_of_mem this.2 (h _ this.1)
      have : (e.symm.trans (swap x (f x) * f)).trans e =
          swap (e x) (e (f x)) * (e.symm.trans f).trans e := by
        ext
        rw [← Equiv.symm_trans_swap_trans, mul_def, Equiv.symm_trans_swap_trans, mul_def]
        repeat (rw [trans_apply])
        simp [swap, swapCore]
        split_ifs <;> rfl
      /-
        case neg
        α : Type u
        inst✝ : DecidableEq α
        n : Nat
        x : α
        l : List α
        f : Equiv.Perm α
        e : Equiv α (Fin n)
        h : ∀ (x_1 : α), Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
        hfx : Not (Eq x (f x))
        hy : ∀ (y : α), Ne ((HMul.hMul (Equiv.swap x (f x)) f) y) y → Membership.mem l y
        this : Eq ((e.symm.trans (HMul.hMul (Equiv.swap x (f x)) f)).trans e) (HMul.hM …
        ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (ite (Eq x (f x)) (Equiv. …
      -/
      have hefx : e x ≠ e (f x) := mt e.injective.eq_iff.1 hfx
      /-
        case neg
        α : Type u
        inst✝ : DecidableEq α
        n : Nat
        x : α
        l : List α
        f : Equiv.Perm α
        e : Equiv α (Fin n)
        h : ∀ (x_1 : α), Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
        hfx : Not (Eq x (f x))
        hy : ∀ (y : α), Ne ((HMul.hMul (Equiv.swap x (f x)) f) y) y → Membership.mem l y
        this : Eq ((e.symm.trans (HMul.hMul (Equiv.swap x (f x)) f)).trans e) (HMul.hM …
        hefx : Ne (e x) (e (f x))
        ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (ite (Eq x (f x)) (Equiv. …
      -/
      rw [if_neg hfx, ← signAux_eq_signAux2 _ _ e hy, this, signAux_mul, signAux_swap hefx]
      /-
        case neg
        α : Type u
        inst✝ : DecidableEq α
        n : Nat
        x : α
        l : List α
        f : Equiv.Perm α
        e : Equiv α (Fin n)
        h : ∀ (x_1 : α), Ne (f x_1) x_1 → Membership.mem (List.cons x l) x_1
        hfx : Not (Eq x (f x))
        hy : ∀ (y : α), Ne ((HMul.hMul (Equiv.swap x (f x)) f) y) y → Membership.mem l y
        this : Eq ((e.symm.trans (HMul.hMul (Equiv.swap x (f x)) f)).trans e) (HMul.hM …
        hefx : Ne (e x) (e (f x))
        ⊢ Eq (Equiv.Perm.signAux ((e.symm.trans f).trans e)) (Neg.neg (HMul.hMul (-1)  …
      -/
      simp only [neg_neg, one_mul, neg_mul]
      /-
        🎉 no goals
      -/


/-- When the multiset `s : Multiset α` contains all nonfixed points of the permutation `f : Perm α`,
  `signAux2 f _` recursively calculates the sign of `f`. -/
def signAux3 [Finite α] (f : Perm α) {s : Multiset α} : (∀ x, x ∈ s) → ℤˣ :=
  Quotient.hrecOn s (fun l _ => signAux2 l f) fun l₁ l₂ h ↦ by
    /-
      α : Type u
      inst✝¹ : DecidableEq α
      β : Type v
      inst✝ : Finite α
      f : Equiv.Perm α
      s : Multiset α
      l₁ l₂ : List α
      h : HasEquiv.Equiv l₁ l₂
      ⊢ HEq (fun x => Equiv.Perm.signAux2 l₁ f) fun x => Equiv.Perm.signAux2 l₂ f
    -/
    rcases Finite.exists_equiv_fin α with ⟨n, ⟨e⟩⟩
    /-
      case intro.intro
      α : Type u
      inst✝¹ : DecidableEq α
      β : Type v
      inst✝ : Finite α
      f : Equiv.Perm α
      s : Multiset α
      l₁ l₂ : List α
      h : HasEquiv.Equiv l₁ l₂
      n : Nat
      e : Equiv α (Fin n)
      ⊢ HEq (fun x => Equiv.Perm.signAux2 l₁ f) fun x => Equiv.Perm.signAux2 l₂ f
    -/
    refine Function.hfunext (forall_congr fun _ ↦ propext h.mem_iff) fun h₁ h₂ _ ↦ ?_
    /-
      case intro.intro
      α : Type u
      inst✝¹ : DecidableEq α
      β : Type v
      inst✝ : Finite α
      f : Equiv.Perm α
      s : Multiset α
      l₁ l₂ : List α
      h : HasEquiv.Equiv l₁ l₂
      n : Nat
      e : Equiv α (Fin n)
      h₁ : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l₁) x
      h₂ : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l₂) x
      x✝ : HEq h₁ h₂
      ⊢ HEq (Equiv.Perm.signAux2 l₁ f) (Equiv.Perm.signAux2 l₂ f)
    -/
    rw [← signAux_eq_signAux2 _ _ e fun _ _ => h₁ _, ← signAux_eq_signAux2 _ _ e fun _ _ => h₂ _]
    /-
      🎉 no goals
    -/


theorem signAux3_mul_and_swap [Finite α] (f g : Perm α) (s : Multiset α) (hs : ∀ x, x ∈ s) :
    signAux3 (f * g) hs = signAux3 f hs * signAux3 g hs ∧
      Pairwise fun x y => signAux3 (swap x y) hs = -1 := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    f g : Equiv.Perm α
    s : Multiset α
    hs : ∀ (x : α), Membership.mem s x
    ⊢ And (Eq ((HMul.hMul f g).signAux3 hs) (HMul.hMul (f.signAux3 hs) (g.signAux3 …
  -/
  obtain ⟨n, ⟨e⟩⟩ := Finite.exists_equiv_fin α
  /-
    case intro.intro
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    f g : Equiv.Perm α
    s : Multiset α
    hs : ∀ (x : α), Membership.mem s x
    n : Nat
    e : Equiv α (Fin n)
    ⊢ And (Eq ((HMul.hMul f g).signAux3 hs) (HMul.hMul (f.signAux3 hs) (g.signAux3 …
  -/
  induction s using Quotient.inductionOn with | _ l => ?_
  show
    signAux2 l (f * g) = signAux2 l f * signAux2 l g ∧
    Pairwise fun x y => signAux2 l (swap x y) = -1
  have hfg : (e.symm.trans (f * g)).trans e = (e.symm.trans f).trans e * (e.symm.trans g).trans e :=
    Equiv.ext fun h => by simp [mul_apply]
  /-
    case intro.intro.h
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    f g : Equiv.Perm α
    n : Nat
    e : Equiv α (Fin n)
    l : List α
    hs : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x
    hfg : Eq ((e.symm.trans (HMul.hMul f g)).trans e) (HMul.hMul ((e.symm.trans f) …
    ⊢ And (Eq (Equiv.Perm.signAux2 l (HMul.hMul f g)) (HMul.hMul (Equiv.Perm.signA …
  -/
  constructor
  · rw [← signAux_eq_signAux2 _ _ e fun _ _ => hs _, ←
      signAux_eq_signAux2 _ _ e fun _ _ => hs _, ← signAux_eq_signAux2 _ _ e fun _ _ => hs _,
      hfg, signAux_mul]
    /-
      case intro.intro.h.right
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      f g : Equiv.Perm α
      n : Nat
      e : Equiv α (Fin n)
      l : List α
      hs : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x
      hfg : Eq ((e.symm.trans (HMul.hMul f g)).trans e) (HMul.hMul ((e.symm.trans f) …
      ⊢ Pairwise fun x y => Eq (Equiv.Perm.signAux2 l (Equiv.swap x y)) (-1)
    -/
  · intro x y hxy
    /-
      case intro.intro.h.right
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      f g : Equiv.Perm α
      n : Nat
      e : Equiv α (Fin n)
      l : List α
      hs : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x
      hfg : Eq ((e.symm.trans (HMul.hMul f g)).trans e) (HMul.hMul ((e.symm.trans f) …
      x y : α
      hxy : Ne x y
      ⊢ Eq (Equiv.Perm.signAux2 l (Equiv.swap x y)) (-1)
    -/
    rw [← e.injective.ne_iff] at hxy
    /-
      case intro.intro.h.right
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Finite α
      f g : Equiv.Perm α
      n : Nat
      e : Equiv α (Fin n)
      l : List α
      hs : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x
      hfg : Eq ((e.symm.trans (HMul.hMul f g)).trans e) (HMul.hMul ((e.symm.trans f) …
      x y : α
      hxy : Ne (e x) (e y)
      ⊢ Eq (Equiv.Perm.signAux2 l (Equiv.swap x y)) (-1)
    -/
    rw [← signAux_eq_signAux2 _ _ e fun _ _ => hs _, symm_trans_swap_trans, signAux_swap hxy]
    /-
      🎉 no goals
    -/


theorem signAux3_symm_trans_trans [Finite α] [DecidableEq β] [Finite β] (f : Perm α) (e : α ≃ β)
    {s : Multiset α} {t : Multiset β} (hs : ∀ x, x ∈ s) (ht : ∀ x, x ∈ t) :
    signAux3 ((e.symm.trans f).trans e) ht = signAux3 f hs := by
  -- Porting note: switched from term mode to tactic mode
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Finite α
    inst✝¹ : DecidableEq β
    inst✝ : Finite β
    f : Equiv.Perm α
    e : Equiv α β
    s : Multiset α
    t : Multiset β
    hs : ∀ (x : α), Membership.mem s x
    ht : ∀ (x : β), Membership.mem t x
    ⊢ Eq (Equiv.Perm.signAux3 ((e.symm.trans f).trans e) ht) (f.signAux3 hs)
  -/
  induction' t, s using Quotient.inductionOn₂ with t s ht hs
  /-
    case h
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Finite α
    inst✝¹ : DecidableEq β
    inst✝ : Finite β
    f : Equiv.Perm α
    e : Equiv α β
    t : List β
    s : List α
    hs : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) s) x
    ht : ∀ (x : β), Membership.mem (Quotient.mk (List.isSetoid β) t) x
    ⊢ Eq (Equiv.Perm.signAux3 ((e.symm.trans f).trans e) ht) (f.signAux3 hs)
  -/
  show signAux2 _ _ = signAux2 _ _
  /-
    case h
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Finite α
    inst✝¹ : DecidableEq β
    inst✝ : Finite β
    f : Equiv.Perm α
    e : Equiv α β
    t : List β
    s : List α
    hs : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) s) x
    ht : ∀ (x : β), Membership.mem (Quotient.mk (List.isSetoid β) t) x
    ⊢ Eq (Equiv.Perm.signAux2 t ((e.symm.trans f).trans e)) (Equiv.Perm.signAux2 s …
  -/
  rcases Finite.exists_equiv_fin β with ⟨n, ⟨e'⟩⟩
  rw [← signAux_eq_signAux2 _ _ e' fun _ _ => ht _,
    ← signAux_eq_signAux2 _ _ (e.trans e') fun _ _ => hs _]
  exact congr_arg signAux
    (Equiv.ext fun x => by simp [Equiv.coe_trans, apply_eq_iff_eq, symm_trans_apply])


/-- `SignType.sign` of a permutation returns the signature or parity of a permutation, `1` for even
permutations, `-1` for odd permutations. It is the unique surjective group homomorphism from
`Perm α` to the group with two elements. -/
def sign [Fintype α] : Perm α →* ℤˣ :=
  MonoidHom.mk' (fun f => signAux3 f mem_univ) fun f g => (signAux3_mul_and_swap f g _ mem_univ).1


@[simp]
theorem sign_mul (f g : Perm α) : sign (f * g) = sign f * sign g :=
  MonoidHom.map_mul sign f g


@[simp]
theorem sign_trans (f g : Perm α) : sign (f.trans g) = sign g * sign f := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    ⊢ Eq (Equiv.Perm.sign (Equiv.trans f g)) (HMul.hMul (Equiv.Perm.sign g) (Equiv …
  -/
  rw [← mul_def, sign_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_one : sign (1 : Perm α) = 1 :=
  MonoidHom.map_one sign


@[simp]
theorem sign_refl : sign (Equiv.refl α) = 1 :=
  MonoidHom.map_one sign


@[simp]
theorem sign_inv (f : Perm α) : sign f⁻¹ = sign f := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Eq (Equiv.Perm.sign (Inv.inv f)) (Equiv.Perm.sign f)
  -/
  rw [MonoidHom.map_inv sign f, Int.units_inv_eq_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_symm (e : Perm α) : sign e.symm = sign e :=
  sign_inv e


theorem sign_swap {x y : α} (h : x ≠ y) : sign (swap x y) = -1 :=
  (signAux3_mul_and_swap 1 1 _ mem_univ).2 h


@[simp]
theorem sign_swap' {x y : α} : sign (swap x y) = if x = y then 1 else -1 :=
                       /-
                         α : Type u
                         inst✝¹ : DecidableEq α
                         inst✝ : Fintype α
                         x y : α
                         H : Eq x y
                         ⊢ Eq (Equiv.Perm.sign (Equiv.swap x y)) (ite (Eq x y) 1 (-1))
                       -/
                       /-
                         🎉 no goals
                       -/
  if H : x = y then by simp [H, swap_self] else by simp [sign_swap H, H]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem IsSwap.sign_eq {f : Perm α} (h : f.IsSwap) : sign f = -1 :=
  let ⟨_, _, hxy⟩ := h
  hxy.2.symm ▸ sign_swap hxy.1


@[simp]
theorem sign_symm_trans_trans [DecidableEq β] [Fintype β] (f : Perm α) (e : α ≃ β) :
    sign ((e.symm.trans f).trans e) = sign f :=
  signAux3_symm_trans_trans f e mem_univ mem_univ


@[simp]
theorem sign_trans_trans_symm [DecidableEq β] [Fintype β] (f : Perm β) (e : α ≃ β) :
    sign ((e.trans f).trans e.symm) = sign f :=
  sign_symm_trans_trans f e.symm


theorem sign_prod_list_swap {l : List (Perm α)} (hl : ∀ g ∈ l, IsSwap g) :
    sign l.prod = (-1) ^ l.length := by
  have h₁ : l.map sign = List.replicate l.length (-1) :=
    List.eq_replicate_iff.2
      ⟨by simp, fun u hu =>
        let ⟨g, hg⟩ := List.mem_map.1 hu
        hg.2 ▸ (hl _ hg.1).sign_eq⟩
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    hl : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap
    h₁ : Eq (List.map (⇑Equiv.Perm.sign) l) (List.replicate l.length (-1))
    ⊢ Eq (Equiv.Perm.sign l.prod) (HPow.hPow (-1) l.length)
  -/
  rw [← List.prod_replicate, ← h₁, List.prod_hom _ (@sign α _ _)]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_abs (f : Perm α) :
    |(Equiv.Perm.sign f : ℤ)| = 1 := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Eq (abs ↑(Equiv.Perm.sign f)) 1
  -/
  rw [Int.abs_eq_natAbs, Int.units_natAbs, Nat.cast_one]
  /-
    🎉 no goals
  -/


theorem sign_surjective [Nontrivial α] : Function.Surjective (sign : Perm α → ℤˣ) := fun a =>
                                                /-
                                                  α : Type u
                                                  inst✝² : DecidableEq α
                                                  inst✝¹ : Fintype α
                                                  inst✝ : Nontrivial α
                                                  a : Units Int
                                                  h : Eq a 1
                                                  ⊢ Eq (Equiv.Perm.sign 1) a
                                                -/
  (Int.units_eq_one_or a).elim (fun h => ⟨1, by simp [h]⟩) fun h =>
                                                /-
                                                  🎉 no goals
                                                -/
    let ⟨x, y, hxy⟩ := exists_pair_ne α
                  /-
                    α : Type u
                    inst✝² : DecidableEq α
                    inst✝¹ : Fintype α
                    inst✝ : Nontrivial α
                    a : Units Int
                    h : Eq a (-1)
                    x y : α
                    hxy : Ne x y
                    ⊢ Eq (Equiv.Perm.sign (Equiv.swap x y)) a
                  -/
    ⟨swap x y, by rw [sign_swap hxy, h]⟩
                  /-
                    🎉 no goals
                  -/


theorem eq_sign_of_surjective_hom {s : Perm α →* ℤˣ} (hs : Surjective s) : s = sign :=
  have : ∀ {f}, IsSwap f → s f = -1 := fun {f} ⟨x, y, hxy, hxy'⟩ =>
    hxy'.symm ▸
      by_contradiction fun h => by
        have : ∀ f, IsSwap f → s f = 1 := fun f ⟨a, b, hab, hab'⟩ => by
          rw [← isConj_iff_eq, ← Or.resolve_right (Int.units_eq_one_or _) h, hab']
          exact s.map_isConj (isConj_swap hab hxy)
        /-
          α : Type u
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          s : MonoidHom (Equiv.Perm α) (Units Int)
          hs : Function.Surjective ⇑s
          f : Equiv.Perm α
          x✝ : f.IsSwap
          x y : α
          hxy : Ne x y
          hxy' : Eq f (Equiv.swap x y)
          h : Not (Eq (s (Equiv.swap x y)) (-1))
          this : ∀ (f : Equiv.Perm α), f.IsSwap → Eq (s f) 1
          ⊢ False
        -/
        let ⟨g, hg⟩ := hs (-1)
        /-
          α : Type u
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          s : MonoidHom (Equiv.Perm α) (Units Int)
          hs : Function.Surjective ⇑s
          f : Equiv.Perm α
          x✝ : f.IsSwap
          x y : α
          hxy : Ne x y
          hxy' : Eq f (Equiv.swap x y)
          h : Not (Eq (s (Equiv.swap x y)) (-1))
          this : ∀ (f : Equiv.Perm α), f.IsSwap → Eq (s f) 1
          g : Equiv.Perm α
          hg : Eq (s g) (-1)
          ⊢ False
        -/
        let ⟨l, hl⟩ := (truncSwapFactors g).out
        have : ∀ a ∈ l.map s, a = (1 : ℤˣ) := fun a ha =>
          let ⟨g, hg⟩ := List.mem_map.1 ha
          hg.2 ▸ this _ (hl.2 _ hg.1)
        have : s l.prod = 1 := by
          rw [← l.prod_hom s, List.eq_replicate_length.2 this, List.prod_replicate, one_pow]
        /-
          α : Type u
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          s : MonoidHom (Equiv.Perm α) (Units Int)
          hs : Function.Surjective ⇑s
          f : Equiv.Perm α
          x✝ : f.IsSwap
          x y : α
          hxy : Ne x y
          hxy' : Eq f (Equiv.swap x y)
          h : Not (Eq (s (Equiv.swap x y)) (-1))
          this✝¹ : ∀ (f : Equiv.Perm α), f.IsSwap → Eq (s f) 1
          g : Equiv.Perm α
          hg : Eq (s g) (-1)
          l : List (Equiv.Perm α)
          hl : And (Eq l.prod g) (∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap)
          this✝ : ∀ (a : Units Int), Membership.mem (List.map (⇑s) l) a → Eq a 1
          this : Eq (s l.prod) 1
          ⊢ False
        -/
        rw [hl.1, hg] at this
        /-
          α : Type u
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          s : MonoidHom (Equiv.Perm α) (Units Int)
          hs : Function.Surjective ⇑s
          f : Equiv.Perm α
          x✝ : f.IsSwap
          x y : α
          hxy : Ne x y
          hxy' : Eq f (Equiv.swap x y)
          h : Not (Eq (s (Equiv.swap x y)) (-1))
          this✝¹ : ∀ (f : Equiv.Perm α), f.IsSwap → Eq (s f) 1
          g : Equiv.Perm α
          hg : Eq (s g) (-1)
          l : List (Equiv.Perm α)
          hl : And (Eq l.prod g) (∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap)
          this✝ : ∀ (a : Units Int), Membership.mem (List.map (⇑s) l) a → Eq a 1
          this : Eq (-1) 1
          ⊢ False
        -/
        exact absurd this (by simp_all)
        /-
          🎉 no goals
        -/
  MonoidHom.ext fun f => by
    /-
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      s : MonoidHom (Equiv.Perm α) (Units Int)
      hs : Function.Surjective ⇑s
      this : ∀ {f : Equiv.Perm α}, f.IsSwap → Eq (s f) (-1)
      f : Equiv.Perm α
      ⊢ Eq (s f) (Equiv.Perm.sign f)
    -/
    let ⟨l, hl₁, hl₂⟩ := (truncSwapFactors f).out
    have hsl : ∀ a ∈ l.map s, a = (-1 : ℤˣ) := fun a ha =>
      let ⟨g, hg⟩ := List.mem_map.1 ha
      hg.2 ▸ this (hl₂ _ hg.1)
    rw [← hl₁, ← l.prod_hom s, List.eq_replicate_length.2 hsl, List.length_map, List.prod_replicate,
      sign_prod_list_swap hl₂]


theorem sign_subtypePerm (f : Perm α) {p : α → Prop} [DecidablePred p] (h₁ : ∀ x, p x ↔ p (f x))
    (h₂ : ∀ x, f x ≠ x → p x) : sign (subtypePerm f h₁) = sign f := by
  /-
    α : Type u
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    f : Equiv.Perm α
    p : α → Prop
    inst✝ : DecidablePred p
    h₁ : ∀ (x : α), Iff (p x) (p (f x))
    h₂ : ∀ (x : α), Ne (f x) x → p x
    ⊢ Eq (Equiv.Perm.sign (f.subtypePerm h₁)) (Equiv.Perm.sign f)
  -/
  let l := (truncSwapFactors (subtypePerm f h₁)).out
  have hl' : ∀ g' ∈ l.1.map ofSubtype, IsSwap g' := fun g' hg' =>
    let ⟨g, hg⟩ := List.mem_map.1 hg'
    hg.2 ▸ (l.2.2 _ hg.1).of_subtype_isSwap
  have hl'₂ : (l.1.map ofSubtype).prod = f := by
    rw [l.1.prod_hom ofSubtype, l.2.1, ofSubtype_subtypePerm _ h₂]
  conv =>
    congr
    rw [← l.2.1]
  /-
    α : Type u
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    f : Equiv.Perm α
    p : α → Prop
    inst✝ : DecidablePred p
    h₁ : ∀ (x : α), Iff (p x) (p (f x))
    h₂ : ∀ (x : α), Ne (f x) x → p x
    l : Subtype fun l => And (Eq l.prod (f.subtypePerm h₁)) (∀ (g : Equiv.Perm (Su …
    hl' : ∀ (g' : Equiv.Perm α), Membership.mem (List.map ⇑Equiv.Perm.ofSubtype ↑l …
    hl'₂ : Eq (List.map ⇑Equiv.Perm.ofSubtype ↑l).prod f
    ⊢ Eq (Equiv.Perm.sign (↑l).prod) (Equiv.Perm.sign f)
  -/
  simp_rw [← hl'₂]
  /-
    α : Type u
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    f : Equiv.Perm α
    p : α → Prop
    inst✝ : DecidablePred p
    h₁ : ∀ (x : α), Iff (p x) (p (f x))
    h₂ : ∀ (x : α), Ne (f x) x → p x
    l : Subtype fun l => And (Eq l.prod (f.subtypePerm h₁)) (∀ (g : Equiv.Perm (Su …
    hl' : ∀ (g' : Equiv.Perm α), Membership.mem (List.map ⇑Equiv.Perm.ofSubtype ↑l …
    hl'₂ : Eq (List.map ⇑Equiv.Perm.ofSubtype ↑l).prod f
    ⊢ Eq (Equiv.Perm.sign (↑l).prod) (Equiv.Perm.sign (List.map ⇑Equiv.Perm.ofSubt …
  -/
  rw [sign_prod_list_swap l.2.2, sign_prod_list_swap hl', List.length_map]
  /-
    🎉 no goals
  -/


theorem sign_eq_sign_of_equiv [DecidableEq β] [Fintype β] (f : Perm α) (g : Perm β) (e : α ≃ β)
    (h : ∀ x, e (f x) = g (e x)) : sign f = sign g := by
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    f : Equiv.Perm α
    g : Equiv.Perm β
    e : Equiv α β
    h : ∀ (x : α), Eq (e (f x)) (g (e x))
    ⊢ Eq (Equiv.Perm.sign f) (Equiv.Perm.sign g)
  -/
  have hg : g = (e.symm.trans f).trans e := Equiv.ext <| by simp [h]
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    f : Equiv.Perm α
    g : Equiv.Perm β
    e : Equiv α β
    h : ∀ (x : α), Eq (e (f x)) (g (e x))
    hg : Eq g ((e.symm.trans f).trans e)
    ⊢ Eq (Equiv.Perm.sign f) (Equiv.Perm.sign g)
  -/
  rw [hg, sign_symm_trans_trans]
  /-
    🎉 no goals
  -/


theorem sign_bij [DecidableEq β] [Fintype β] {f : Perm α} {g : Perm β} (i : ∀ x : α, f x ≠ x → β)
    (h : ∀ x hx hx', i (f x) hx' = g (i x hx)) (hi : ∀ x₁ x₂ hx₁ hx₂, i x₁ hx₁ = i x₂ hx₂ → x₁ = x₂)
    (hg : ∀ y, g y ≠ y → ∃ x hx, i x hx = y) : sign f = sign g :=
  calc
                                       /-
                                         α : Type u
                                         inst✝³ : DecidableEq α
                                         β : Type v
                                         inst✝² : Fintype α
                                         inst✝¹ : DecidableEq β
                                         inst✝ : Fintype β
                                         f : Equiv.Perm α
                                         g : Equiv.Perm β
                                         i : (x : α) → Ne (f x) x → β
                                         h : ∀ (x : α) (hx : Ne (f x) x) (hx' : Ne (f (f x)) (f x)), Eq (i (f x) hx') ( …
                                         hi : ∀ (x₁ x₂ : α) (hx₁ : Ne (f x₁) x₁) (hx₂ : Ne (f x₂) x₂), Eq (i x₁ hx₁) (i …
                                         hg : ∀ (y : β), Ne (g y) y → Exists fun x => Exists fun hx => Eq (i x hx) y
                                         ⊢ ∀ (x : α), Iff (Ne (f x) x) (Ne (f (f x)) (f x))
                                       -/
    sign f = sign (subtypePerm f <| by simp : Perm { x // f x ≠ x }) :=
                                       /-
                                         🎉 no goals
                                       -/
      (sign_subtypePerm _ _ fun _ => id).symm
                                  /-
                                    α : Type u
                                    inst✝³ : DecidableEq α
                                    β : Type v
                                    inst✝² : Fintype α
                                    inst✝¹ : DecidableEq β
                                    inst✝ : Fintype β
                                    f : Equiv.Perm α
                                    g : Equiv.Perm β
                                    i : (x : α) → Ne (f x) x → β
                                    h : ∀ (x : α) (hx : Ne (f x) x) (hx' : Ne (f (f x)) (f x)), Eq (i (f x) hx') ( …
                                    hi : ∀ (x₁ x₂ : α) (hx₁ : Ne (f x₁) x₁) (hx₂ : Ne (f x₂) x₂), Eq (i x₁ hx₁) (i …
                                    hg : ∀ (y : β), Ne (g y) y → Exists fun x => Exists fun hx => Eq (i x hx) y
                                    ⊢ ∀ (x : β), Iff (Ne (g x) x) (Ne (g (g x)) (g x))
                                  -/
    _ = sign (subtypePerm g <| by simp : Perm { x // g x ≠ x }) :=
                                  /-
                                    🎉 no goals
                                  -/
      sign_eq_sign_of_equiv _ _
        (Equiv.ofBijective
          (fun x : { x // f x ≠ x } =>
            (⟨i x.1 x.2, by
                /-
                  α : Type u
                  inst✝³ : DecidableEq α
                  β : Type v
                  inst✝² : Fintype α
                  inst✝¹ : DecidableEq β
                  inst✝ : Fintype β
                  f : Equiv.Perm α
                  g : Equiv.Perm β
                  i : (x : α) → Ne (f x) x → β
                  h : ∀ (x : α) (hx : Ne (f x) x) (hx' : Ne (f (f x)) (f x)), Eq (i (f x) hx') ( …
                  hi : ∀ (x₁ x₂ : α) (hx₁ : Ne (f x₁) x₁) (hx₂ : Ne (f x₂) x₂), Eq (i x₁ hx₁) (i …
                  hg : ∀ (y : β), Ne (g y) y → Exists fun x => Exists fun hx => Eq (i x hx) y
                  x : Subtype fun x => Ne (f x) x
                  ⊢ Ne (g (i ↑x ⋯)) (i ↑x ⋯)
                -/
                have : f (f x) ≠ f x := mt (fun h => f.injective h) x.2
                /-
                  α : Type u
                  inst✝³ : DecidableEq α
                  β : Type v
                  inst✝² : Fintype α
                  inst✝¹ : DecidableEq β
                  inst✝ : Fintype β
                  f : Equiv.Perm α
                  g : Equiv.Perm β
                  i : (x : α) → Ne (f x) x → β
                  h : ∀ (x : α) (hx : Ne (f x) x) (hx' : Ne (f (f x)) (f x)), Eq (i (f x) hx') ( …
                  hi : ∀ (x₁ x₂ : α) (hx₁ : Ne (f x₁) x₁) (hx₂ : Ne (f x₂) x₂), Eq (i x₁ hx₁) (i …
                  hg : ∀ (y : β), Ne (g y) y → Exists fun x => Exists fun hx => Eq (i x hx) y
                  x : Subtype fun x => Ne (f x) x
                  this : Ne (f (f ↑x)) (f ↑x)
                  ⊢ Ne (g (i ↑x ⋯)) (i ↑x ⋯)
                -/
                rw [← h _ x.2 this]
                /-
                  α : Type u
                  inst✝³ : DecidableEq α
                  β : Type v
                  inst✝² : Fintype α
                  inst✝¹ : DecidableEq β
                  inst✝ : Fintype β
                  f : Equiv.Perm α
                  g : Equiv.Perm β
                  i : (x : α) → Ne (f x) x → β
                  h : ∀ (x : α) (hx : Ne (f x) x) (hx' : Ne (f (f x)) (f x)), Eq (i (f x) hx') ( …
                  hi : ∀ (x₁ x₂ : α) (hx₁ : Ne (f x₁) x₁) (hx₂ : Ne (f x₂) x₂), Eq (i x₁ hx₁) (i …
                  hg : ∀ (y : β), Ne (g y) y → Exists fun x => Exists fun hx => Eq (i x hx) y
                  x : Subtype fun x => Ne (f x) x
                  this : Ne (f (f ↑x)) (f ↑x)
                  ⊢ Ne (i (f ↑x) this) (i ↑x ⋯)
                -/
                exact mt (hi _ _ this x.2) x.2⟩ :
                /-
                  🎉 no goals
                -/
              { y // g y ≠ y }))
          ⟨fun ⟨_, _⟩ ⟨_, _⟩ h => Subtype.eq (hi _ _ _ _ (Subtype.mk.inj h)), fun ⟨y, hy⟩ =>
            let ⟨x, hfx, hx⟩ := hg y hy
            ⟨⟨x, hfx⟩, Subtype.eq hx⟩⟩)
        fun ⟨x, _⟩ => Subtype.eq (h x _ _)
    _ = sign g := sign_subtypePerm _ _ fun _ => id


/-- If we apply `prod_extendRight a (σ a)` for all `a : α` in turn,
we get `prod_congrRight σ`. -/
theorem prod_prodExtendRight {α : Type*} [DecidableEq α] (σ : α → Perm β) {l : List α}
    (hl : l.Nodup) (mem_l : ∀ a, a ∈ l) :
    (l.map fun a => prodExtendRight a (σ a)).prod = prodCongrRight σ := by
  /-
    β : Type v
    α : Type u_1
    inst✝ : DecidableEq α
    σ : α → Equiv.Perm β
    l : List α
    hl : l.Nodup
    mem_l : ∀ (a : α), Membership.mem l a
    ⊢ Eq (List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod (Equiv.pr …
  -/
  ext ⟨a, b⟩ : 1
  -- We'll use induction on the list of elements,
  -- but we have to keep track of whether we already passed `a` in the list.
  suffices a ∈ l ∧ (l.map fun a => prodExtendRight a (σ a)).prod (a, b) = (a, σ a b) ∨
      a ∉ l ∧ (l.map fun a => prodExtendRight a (σ a)).prod (a, b) = (a, b) by
    obtain ⟨_, prod_eq⟩ := Or.resolve_right this (not_and.mpr fun h _ => h (mem_l a))
    rw [prod_eq, prodCongrRight_apply]
  /-
    case H.mk
    β : Type v
    α : Type u_1
    inst✝ : DecidableEq α
    σ : α → Equiv.Perm β
    l : List α
    hl : l.Nodup
    mem_l : ∀ (a : α), Membership.mem l a
    a : α
    b : β
    ⊢ Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Perm.prodExtendR …
  -/
  clear mem_l
  /-
    case H.mk
    β : Type v
    α : Type u_1
    inst✝ : DecidableEq α
    σ : α → Equiv.Perm β
    l : List α
    hl : l.Nodup
    a : α
    b : β
    ⊢ Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Perm.prodExtendR …
  -/
  induction' l with a' l ih
    /-
      case H.mk.nil
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      hl : List.nil.Nodup
      ⊢ Or (And (Membership.mem List.nil a) (Eq ((List.map (fun a => Equiv.Perm.prod …
    -/
  · refine Or.inr ⟨List.not_mem_nil _, ?_⟩
    /-
      case H.mk.nil
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      hl : List.nil.Nodup
      ⊢ Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) List.nil).prod { …
    -/
    rw [List.map_nil, List.prod_nil, one_apply]
    /-
      🎉 no goals
    -/
  /-
    case H.mk.cons
    β : Type v
    α : Type u_1
    inst✝ : DecidableEq α
    σ : α → Equiv.Perm β
    a : α
    b : β
    a' : α
    l : List α
    ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
    hl : (List.cons a' l).Nodup
    ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((List.map (fun a => Equiv.P …
  -/
  rw [List.map_cons, List.prod_cons, mul_apply]
  /-
    case H.mk.cons
    β : Type v
    α : Type u_1
    inst✝ : DecidableEq α
    σ : α → Equiv.Perm β
    a : α
    b : β
    a' : α
    l : List α
    ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
    hl : (List.cons a' l).Nodup
    ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((Equiv.Perm.prodExtendRight …
  -/
  rcases ih (List.nodup_cons.mp hl).2 with (⟨mem_l, prod_eq⟩ | ⟨not_mem_l, prod_eq⟩) <;>
    /-
      case H.mk.cons.inl.intro
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      mem_l : Membership.mem l a
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((Equiv.Perm.prodExtendRight …
    -/
    rw [prod_eq]
    /-
      case H.mk.cons.inl.intro
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      mem_l : Membership.mem l a
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((Equiv.Perm.prodExtendRight …
    -/
  · refine Or.inl ⟨List.mem_cons_of_mem _ mem_l, ?_⟩
    /-
      case H.mk.cons.inl.intro
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      mem_l : Membership.mem l a
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ⊢ Eq ((Equiv.Perm.prodExtendRight a' (σ a')) { fst := a, snd := (σ a) b }) { f …
    -/
    rw [prodExtendRight_apply_ne _ fun h : a = a' => (List.nodup_cons.mp hl).1 (h ▸ mem_l)]
    /-
      🎉 no goals
    -/
  /-
    case H.mk.cons.inr.intro
    β : Type v
    α : Type u_1
    inst✝ : DecidableEq α
    σ : α → Equiv.Perm β
    a : α
    b : β
    a' : α
    l : List α
    ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
    hl : (List.cons a' l).Nodup
    not_mem_l : Not (Membership.mem l a)
    prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
    ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((Equiv.Perm.prodExtendRight …
  -/
  by_cases ha' : a = a'
    /-
      case pos
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      not_mem_l : Not (Membership.mem l a)
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ha' : Eq a a'
      ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((Equiv.Perm.prodExtendRight …
    -/
  · rw [← ha'] at *
    /-
      case pos
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      not_mem_l : Not (Membership.mem l a)
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ha' : Eq a a
      ⊢ Or (And (Membership.mem (List.cons a l) a) (Eq ((Equiv.Perm.prodExtendRight  …
    -/
    refine Or.inl ⟨l.mem_cons_self a, ?_⟩
    /-
      case pos
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      not_mem_l : Not (Membership.mem l a)
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ha' : Eq a a
      ⊢ Eq ((Equiv.Perm.prodExtendRight a (σ a)) { fst := a, snd := b }) { fst := a, …
    -/
    rw [prodExtendRight_apply_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      not_mem_l : Not (Membership.mem l a)
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ha' : Not (Eq a a')
      ⊢ Or (And (Membership.mem (List.cons a' l) a) (Eq ((Equiv.Perm.prodExtendRight …
    -/
  · refine Or.inr ⟨fun h => not_or_intro ha' not_mem_l ((List.mem_cons).mp h), ?_⟩
    /-
      case neg
      β : Type v
      α : Type u_1
      inst✝ : DecidableEq α
      σ : α → Equiv.Perm β
      a : α
      b : β
      a' : α
      l : List α
      ih : l.Nodup → Or (And (Membership.mem l a) (Eq ((List.map (fun a => Equiv.Per …
      hl : (List.cons a' l).Nodup
      not_mem_l : Not (Membership.mem l a)
      prod_eq : Eq ((List.map (fun a => Equiv.Perm.prodExtendRight a (σ a)) l).prod  …
      ha' : Not (Eq a a')
      ⊢ Eq ((Equiv.Perm.prodExtendRight a' (σ a')) { fst := a, snd := b }) { fst :=  …
    -/
    rw [prodExtendRight_apply_ne _ ha']
    /-
      🎉 no goals
    -/


@[simp]
theorem sign_prodExtendRight (a : α) (σ : Perm β) : sign (prodExtendRight a σ) = sign σ :=
  sign_bij (fun (ab : α × β) _ => ab.snd)
                             /-
                               α : Type u
                               inst✝³ : DecidableEq α
                               β : Type v
                               inst✝² : Fintype α
                               inst✝¹ : DecidableEq β
                               inst✝ : Fintype β
                               a : α
                               σ : Equiv.Perm β
                               x✝¹ : Prod α β
                               a' : α
                               b : β
                               hab : Ne ((Equiv.Perm.prodExtendRight a σ) { fst := a', snd := b }) { fst := a …
                               x✝ : Ne ((Equiv.Perm.prodExtendRight a σ) ((Equiv.Perm.prodExtendRight a σ) {  …
                               ⊢ Eq ((fun ab x => ab.2) ((Equiv.Perm.prodExtendRight a σ) { fst := a', snd := …
                             -/
    (fun ⟨a', b⟩ hab _ => by simp [eq_of_prodExtendRight_ne hab])
                             /-
                               🎉 no goals
                             -/
    (fun ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ hab₁ hab₂ h => by
      /-
        α : Type u
        inst✝³ : DecidableEq α
        β : Type v
        inst✝² : Fintype α
        inst✝¹ : DecidableEq β
        inst✝ : Fintype β
        a : α
        σ : Equiv.Perm β
        x✝¹ x✝ : Prod α β
        a₁ : α
        b₁ : β
        hab₁ : Ne ((Equiv.Perm.prodExtendRight a σ) { fst := a₁, snd := b₁ }) { fst := …
        a₂ : α
        b₂ : β
        hab₂ : Ne ((Equiv.Perm.prodExtendRight a σ) { fst := a₂, snd := b₂ }) { fst := …
        h : Eq ((fun ab x => ab.2) { fst := a₁, snd := b₁ } hab₁) ((fun ab x => ab.2)  …
        ⊢ Eq { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
      -/
      simpa [eq_of_prodExtendRight_ne hab₁, eq_of_prodExtendRight_ne hab₂] using h)
      /-
        🎉 no goals
      -/
                            /-
                              α : Type u
                              inst✝³ : DecidableEq α
                              β : Type v
                              inst✝² : Fintype α
                              inst✝¹ : DecidableEq β
                              inst✝ : Fintype β
                              a : α
                              σ : Equiv.Perm β
                              y : β
                              hy : Ne (σ y) y
                              ⊢ Ne ((Equiv.Perm.prodExtendRight a σ) { fst := a, snd := y }) { fst := a, snd …
                            -/
                            /-
                              🎉 no goals
                            -/
    fun y hy => ⟨(a, y), by simpa, by simp⟩
                                      /-
                                        🎉 no goals
                                      -/


theorem sign_prodCongrRight (σ : α → Perm β) : sign (prodCongrRight σ) = ∏ k, sign (σ k) := by
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    σ : α → Equiv.Perm β
    ⊢ Eq (Equiv.Perm.sign (Equiv.prodCongrRight σ)) (Finset.univ.prod fun k => Equ …
  -/
  obtain ⟨l, hl, mem_l⟩ := Finite.exists_univ_list α
  have l_to_finset : l.toFinset = Finset.univ := by
    apply eq_top_iff.mpr
    intro b _
    exact List.mem_toFinset.mpr (mem_l b)
  rw [← prod_prodExtendRight σ hl mem_l, map_list_prod sign, List.map_map, ← l_to_finset,
    List.prod_toFinset _ hl]
  /-
    case intro.intro
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    σ : α → Equiv.Perm β
    l : List α
    hl : l.Nodup
    mem_l : ∀ (x : α), Membership.mem l x
    l_to_finset : Eq l.toFinset Finset.univ
    ⊢ Eq (List.map (Function.comp ⇑Equiv.Perm.sign fun a => Equiv.Perm.prodExtendR …
  -/
  simp_rw [← fun a => sign_prodExtendRight a (σ a), Function.comp_def]
  /-
    🎉 no goals
  -/


theorem sign_prodCongrLeft (σ : α → Perm β) : sign (prodCongrLeft σ) = ∏ k, sign (σ k) := by
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    σ : α → Equiv.Perm β
    ⊢ Eq (Equiv.Perm.sign (Equiv.prodCongrLeft σ)) (Finset.univ.prod fun k => Equi …
  -/
  refine (sign_eq_sign_of_equiv _ _ (prodComm β α) ?_).trans (sign_prodCongrRight σ)
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    σ : α → Equiv.Perm β
    ⊢ ∀ (x : Prod β α), Eq ((Equiv.prodComm β α) ((Equiv.prodCongrLeft σ) x)) ((Eq …
  -/
  rintro ⟨b, α⟩
  /-
    case mk
    α✝ : Type u
    inst✝³ : DecidableEq α✝
    β : Type v
    inst✝² : Fintype α✝
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    σ : α✝ → Equiv.Perm β
    b : β
    α : α✝
    ⊢ Eq ((Equiv.prodComm β α✝) ((Equiv.prodCongrLeft σ) { fst := b, snd := α }))  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_permCongr (e : α ≃ β) (p : Perm α) : sign (e.permCongr p) = sign p :=
                                       /-
                                         α : Type u
                                         inst✝³ : DecidableEq α
                                         β : Type v
                                         inst✝² : Fintype α
                                         inst✝¹ : DecidableEq β
                                         inst✝ : Fintype β
                                         e : Equiv α β
                                         p : Equiv.Perm α
                                         ⊢ ∀ (x : β), Eq (e.symm ((e.permCongr p) x)) (p (e.symm x))
                                       -/
  sign_eq_sign_of_equiv _ _ e.symm (by simp)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem sign_sumCongr (σa : Perm α) (σb : Perm β) : sign (sumCongr σa σb) = sign σa * sign σb := by
  suffices sign (sumCongr σa (1 : Perm β)) = sign σa ∧ sign (sumCongr (1 : Perm α) σb) = sign σb
    by rw [← this.1, ← this.2, ← sign_mul, sumCongr_mul, one_mul, mul_one]
  /-
    α : Type u
    inst✝³ : DecidableEq α
    β : Type v
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    inst✝ : Fintype β
    σa : Equiv.Perm α
    σb : Equiv.Perm β
    ⊢ And (Eq (Equiv.Perm.sign (σa.sumCongr 1)) (Equiv.Perm.sign σa)) (Eq (Equiv.P …
  -/
  constructor
    /-
      case left
      α : Type u
      inst✝³ : DecidableEq α
      β : Type v
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      inst✝ : Fintype β
      σa : Equiv.Perm α
      σb : Equiv.Perm β
      ⊢ Eq (Equiv.Perm.sign (σa.sumCongr 1)) (Equiv.Perm.sign σa)
    -/
  · refine σa.swap_induction_on ?_ fun σa' a₁ a₂ ha ih => ?_
      /-
        case left.refine_1
        α : Type u
        inst✝³ : DecidableEq α
        β : Type v
        inst✝² : Fintype α
        inst✝¹ : DecidableEq β
        inst✝ : Fintype β
        σa : Equiv.Perm α
        σb : Equiv.Perm β
        ⊢ Eq (Equiv.Perm.sign (Equiv.Perm.sumCongr 1 1)) (Equiv.Perm.sign 1)
      -/
    · simp
      /-
        🎉 no goals
      -/
    · rw [← one_mul (1 : Perm β), ← sumCongr_mul, sign_mul, sign_mul, ih, sumCongr_swap_one,
        sign_swap ha, sign_swap (Sum.inl_injective.ne_iff.mpr ha)]
    /-
      case right
      α : Type u
      inst✝³ : DecidableEq α
      β : Type v
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      inst✝ : Fintype β
      σa : Equiv.Perm α
      σb : Equiv.Perm β
      ⊢ Eq (Equiv.Perm.sign (Equiv.Perm.sumCongr 1 σb)) (Equiv.Perm.sign σb)
    -/
  · refine σb.swap_induction_on ?_ fun σb' b₁ b₂ hb ih => ?_
      /-
        case right.refine_1
        α : Type u
        inst✝³ : DecidableEq α
        β : Type v
        inst✝² : Fintype α
        inst✝¹ : DecidableEq β
        inst✝ : Fintype β
        σa : Equiv.Perm α
        σb : Equiv.Perm β
        ⊢ Eq (Equiv.Perm.sign (Equiv.Perm.sumCongr 1 1)) (Equiv.Perm.sign 1)
      -/
    · simp
      /-
        🎉 no goals
      -/
    · rw [← one_mul (1 : Perm α), ← sumCongr_mul, sign_mul, sign_mul, ih, sumCongr_one_swap,
        sign_swap hb, sign_swap (Sum.inr_injective.ne_iff.mpr hb)]


@[simp]
theorem sign_subtypeCongr {p : α → Prop} [DecidablePred p] (ep : Perm { a // p a })
    (en : Perm { a // ¬p a }) : sign (ep.subtypeCongr en) = sign ep * sign en := by
  /-
    α : Type u
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    ep : Equiv.Perm (Subtype fun a => p a)
    en : Equiv.Perm (Subtype fun a => Not (p a))
    ⊢ Eq (Equiv.Perm.sign (ep.subtypeCongr en)) (HMul.hMul (Equiv.Perm.sign ep) (E …
  -/
  simp [subtypeCongr]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_extendDomain (e : Perm α) {p : β → Prop} [DecidablePred p] (f : α ≃ Subtype p) :
    Equiv.Perm.sign (e.extendDomain f) = Equiv.Perm.sign e := by
  /-
    α : Type u
    inst✝⁴ : DecidableEq α
    β : Type v
    inst✝³ : Fintype α
    inst✝² : DecidableEq β
    inst✝¹ : Fintype β
    e : Equiv.Perm α
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    ⊢ Eq (Equiv.Perm.sign (e.extendDomain f)) (Equiv.Perm.sign e)
  -/
  simp only [Equiv.Perm.extendDomain, sign_subtypeCongr, sign_permCongr, sign_refl, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_ofSubtype {p : α → Prop} [DecidablePred p] (f : Equiv.Perm (Subtype p)) :
    sign (ofSubtype f) = sign f :=
  sign_extendDomain f (Equiv.refl (Subtype p))


/-- Permutations of a given sign. -/
def ofSign (s : ℤˣ) : Finset (Perm α) := univ.filter (sign · = s)


@[simp]
lemma mem_ofSign {s : ℤˣ} {σ : Perm α} : σ ∈ ofSign s ↔ σ.sign = s := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Units Int
    σ : Equiv.Perm α
    ⊢ Iff (Membership.mem (Equiv.Perm.ofSign s) σ) (Eq (Equiv.Perm.sign σ) s)
  -/
  rw [ofSign, mem_filter, and_iff_right (mem_univ σ)]
  /-
    🎉 no goals
  -/


lemma ofSign_disjoint : _root_.Disjoint (ofSign 1 : Finset (Perm α)) (ofSign (-1)) := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ _root_.Disjoint (Equiv.Perm.ofSign 1) (Equiv.Perm.ofSign (-1))
  -/
  rw [Finset.disjoint_iff_ne]
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ ∀ (a : Equiv.Perm α), Membership.mem (Equiv.Perm.ofSign 1) a → ∀ (b : Equiv. …
  -/
  rintro σ hσ τ hτ rfl
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    hσ : Membership.mem (Equiv.Perm.ofSign 1) σ
    hτ : Membership.mem (Equiv.Perm.ofSign (-1)) σ
    ⊢ False
  -/
  rw [mem_ofSign] at hσ hτ
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    hσ : Eq (Equiv.Perm.sign σ) 1
    hτ : Eq (Equiv.Perm.sign σ) (-1)
    ⊢ False
  -/
  have := hσ.symm.trans hτ
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    hσ : Eq (Equiv.Perm.sign σ) 1
    hτ : Eq (Equiv.Perm.sign σ) (-1)
    this : Eq 1 (-1)
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


lemma ofSign_disjUnion :
    (ofSign 1).disjUnion (ofSign (-1)) ofSign_disjoint = (univ : Finset (Perm α)) := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq ((Equiv.Perm.ofSign 1).disjUnion (Equiv.Perm.ofSign (-1)) ⋯) Finset.univ
  -/
  ext σ
  /-
    case h
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    ⊢ Iff (Membership.mem ((Equiv.Perm.ofSign 1).disjUnion (Equiv.Perm.ofSign (-1) …
  -/
  simp_rw [mem_disjUnion, mem_ofSign, Int.units_eq_one_or, mem_univ]
  /-
    🎉 no goals
  -/


