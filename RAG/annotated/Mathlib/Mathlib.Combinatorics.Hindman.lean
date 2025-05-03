/-- Multiplication of ultrafilters given by `∀ᶠ m in U*V, p m ↔ ∀ᶠ m in U, ∀ᶠ m' in V, p (m*m')`. -/
@[to_additive
      "Addition of ultrafilters given by `∀ᶠ m in U+V, p m ↔ ∀ᶠ m in U, ∀ᶠ m' in V, p (m+m')`."]
def Ultrafilter.mul {M} [Mul M] : Mul (Ultrafilter M) where mul U V := (· * ·) <$> U <*> V


@[to_additive]
theorem Ultrafilter.eventually_mul {M} [Mul M] (U V : Ultrafilter M) (p : M → Prop) :
    (∀ᶠ m in ↑(U * V), p m) ↔ ∀ᶠ m in U, ∀ᶠ m' in V, p (m * m') :=
  Iff.rfl


/-- Semigroup structure on `Ultrafilter M` induced by a semigroup structure on `M`. -/
@[to_additive
      "Additive semigroup structure on `Ultrafilter M` induced by an additive semigroup
      structure on `M`."]
def Ultrafilter.semigroup {M} [Semigroup M] : Semigroup (Ultrafilter M) :=
  { Ultrafilter.mul with
    mul_assoc := fun U V W =>
      Ultrafilter.coe_inj.mp <|
                                /-
                                  M : Type ?u.654
                                  inst✝ : Semigroup M
                                  U V W : Ultrafilter M
                                  p : M → Prop
                                  ⊢ Iff (Filter.Eventually (fun x => p x) ↑(HMul.hMul (HMul.hMul U V) W)) (Filte …
                                -/
        Filter.ext' fun p => by simp [Ultrafilter.eventually_mul, mul_assoc] }
                                /-
                                  🎉 no goals
                                -/


@[to_additive]
theorem Ultrafilter.continuous_mul_left {M} [Semigroup M] (V : Ultrafilter M) :
    Continuous (· * V) :=
  ultrafilterBasis_is_basis.continuous_iff.2 <| Set.forall_mem_range.mpr fun s ↦
    ultrafilter_isOpen_basic { m : M | ∀ᶠ m' in V, m * m' ∈ s }


/-- `FS a` is the set of finite sums in `a`, i.e. `m ∈ FS a` if `m` is the sum of a nonempty
subsequence of `a`. We give a direct inductive definition instead of talking about subsequences. -/
inductive FS {M} [AddSemigroup M] : Stream' M → Set M
  | head (a : Stream' M) : FS a a.head
  | tail (a : Stream' M) (m : M) (h : FS a.tail m) : FS a m
  | cons (a : Stream' M) (m : M) (h : FS a.tail m) : FS a (a.head + m)


/-- `FP a` is the set of finite products in `a`, i.e. `m ∈ FP a` if `m` is the product of a nonempty
subsequence of `a`. We give a direct inductive definition instead of talking about subsequences. -/
@[to_additive FS]
inductive FP {M} [Semigroup M] : Stream' M → Set M
  | head (a : Stream' M) : FP a a.head
  | tail (a : Stream' M) (m : M) (h : FP a.tail m) : FP a m
  | cons (a : Stream' M) (m : M) (h : FP a.tail m) : FP a (a.head * m)


/-- If `m` and `m'` are finite products in `M`, then so is `m * m'`, provided that `m'` is obtained
from a subsequence of `M` starting sufficiently late. -/
@[to_additive
      "If `m` and `m'` are finite sums in `M`, then so is `m + m'`, provided that `m'`
      is obtained from a subsequence of `M` starting sufficiently late."]
theorem FP.mul {M} [Semigroup M] {a : Stream' M} {m : M} (hm : m ∈ FP a) :
    ∃ n, ∀ m' ∈ FP (a.drop n), m * m' ∈ FP a := by
  /-
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    m : M
    hm : Membership.mem (Hindman.FP a) m
    ⊢ Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a)) m …
  -/
  induction' hm with a a m hm ih a m hm ih
    /-
      case head
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m : M
      a : Stream' M
      ⊢ Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a)) m …
    -/
  · exact ⟨1, fun m hm => FP.cons a m hm⟩
    /-
      🎉 no goals
    -/
    /-
      case tail
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      ih : Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a. …
      ⊢ Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a)) m …
    -/
  · cases' ih with n hn
    /-
      case tail.intro
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      ⊢ Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a)) m …
    -/
    use n + 1
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      ⊢ ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n 1) a)) m'  …
    -/
    intro m' hm'
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      m' : M
      hm' : Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n 1) a)) m'
      ⊢ Membership.mem (Hindman.FP a) (HMul.hMul m m')
    -/
    exact FP.tail _ _ (hn _ hm')
    /-
      🎉 no goals
    -/
    /-
      case cons
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      ih : Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a. …
      ⊢ Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a)) m …
    -/
  · cases' ih with n hn
    /-
      case cons.intro
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      ⊢ Exists fun n => ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a)) m …
    -/
    use n + 1
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      ⊢ ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n 1) a)) m'  …
    -/
    intro m' hm'
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      m' : M
      hm' : Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n 1) a)) m'
      ⊢ Membership.mem (Hindman.FP a) (HMul.hMul (HMul.hMul a.head m) m')
    -/
    rw [mul_assoc]
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a✝ : Stream' M
      m✝ : M
      a : Stream' M
      m : M
      hm : Hindman.FP a.tail m
      n : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n a.tail)) m' → Memb …
      m' : M
      hm' : Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n 1) a)) m'
      ⊢ Membership.mem (Hindman.FP a) (HMul.hMul a.head (HMul.hMul m m'))
    -/
    exact FP.cons _ _ (hn _ hm')
    /-
      🎉 no goals
    -/


@[to_additive exists_idempotent_ultrafilter_le_FS]
theorem exists_idempotent_ultrafilter_le_FP {M} [Semigroup M] (a : Stream' M) :
    ∃ U : Ultrafilter M, U * U = U ∧ ∀ᶠ m in U, m ∈ FP a := by
  /-
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    ⊢ Exists fun U => And (Eq (HMul.hMul U U) U) (Filter.Eventually (fun m => Memb …
  -/
  let S : Set (Ultrafilter M) := ⋂ n, { U | ∀ᶠ m in U, m ∈ FP (a.drop n) }
  /-
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
    ⊢ Exists fun U => And (Eq (HMul.hMul U U) U) (Filter.Eventually (fun m => Memb …
  -/
  have h := exists_idempotent_in_compact_subsemigroup ?_ S ?_ ?_ ?_
    /-
      case refine_5
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      h : Exists fun m => And (Membership.mem S m) (Eq (HMul.hMul m m) m)
      ⊢ Exists fun U => And (Eq (HMul.hMul U U) U) (Filter.Eventually (fun m => Memb …
    -/
  · rcases h with ⟨U, hU, U_idem⟩
    /-
      case refine_5.intro.intro
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : Membership.mem S U
      U_idem : Eq (HMul.hMul U U) U
      ⊢ Exists fun U => And (Eq (HMul.hMul U U) U) (Filter.Eventually (fun m => Memb …
    -/
    refine ⟨U, U_idem, ?_⟩
    /-
      case refine_5.intro.intro
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : Membership.mem S U
      U_idem : Eq (HMul.hMul U U) U
      ⊢ Filter.Eventually (fun m => Membership.mem (Hindman.FP a) m) ↑U
    -/
    convert Set.mem_iInter.mp hU 0
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      ⊢ ∀ (r : Ultrafilter M), Continuous fun x => HMul.hMul x r
    -/
  · exact Ultrafilter.continuous_mul_left
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      ⊢ S.Nonempty
    -/
  · apply IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed
      /-
        case refine_2.htd
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        ⊢ ∀ (i : Nat), HasSubset.Subset (setOf fun U => Filter.Eventually (fun m => Me …
      -/
    · intro n U hU
      /-
        case refine_2.htd
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        n : Nat
        U : Ultrafilter M
        hU : Membership.mem (setOf fun U => Filter.Eventually (fun m => Membership.mem …
        ⊢ Membership.mem (setOf fun U => Filter.Eventually (fun m => Membership.mem (H …
      -/
      filter_upwards [hU]
      /-
        case h
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        n : Nat
        U : Ultrafilter M
        hU : Membership.mem (setOf fun U => Filter.Eventually (fun m => Membership.mem …
        ⊢ ∀ (a_1 : M), Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n 1) a)) a_ …
      -/
      rw [add_comm, ← Stream'.drop_drop, ← Stream'.tail_eq_drop]
      /-
        case h
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        n : Nat
        U : Ultrafilter M
        hU : Membership.mem (setOf fun U => Filter.Eventually (fun m => Membership.mem …
        ⊢ ∀ (a_1 : M), Membership.mem (Hindman.FP (Stream'.drop n a).tail) a_1 → Membe …
      -/
      exact FP.tail _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.htn
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        ⊢ ∀ (i : Nat), (setOf fun U => Filter.Eventually (fun m => Membership.mem (Hin …
      -/
    · intro n
      /-
        case refine_2.htn
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        n : Nat
        ⊢ (setOf fun U => Filter.Eventually (fun m => Membership.mem (Hindman.FP (Stre …
      -/
      exact ⟨pure _, mem_pure.mpr <| FP.head _⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.ht0
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        ⊢ IsCompact (setOf fun U => Filter.Eventually (fun m => Membership.mem (Hindma …
      -/
    · exact (ultrafilter_isClosed_basic _).isCompact
      /-
        🎉 no goals
      -/
      /-
        case refine_2.htcl
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        ⊢ ∀ (i : Nat), IsClosed (setOf fun U => Filter.Eventually (fun m => Membership …
      -/
    · intro n
      /-
        case refine_2.htcl
        M : Type u_1
        inst✝ : Semigroup M
        a : Stream' M
        S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
        n : Nat
        ⊢ IsClosed (setOf fun U => Filter.Eventually (fun m => Membership.mem (Hindman …
      -/
      apply ultrafilter_isClosed_basic
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      ⊢ IsCompact S
    -/
  · exact IsClosed.isCompact (isClosed_iInter fun i => ultrafilter_isClosed_basic _)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      ⊢ ∀ (x : Ultrafilter M), Membership.mem S x → ∀ (y : Ultrafilter M), Membershi …
    -/
  · intro U hU V hV
    /-
      case refine_4
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : Membership.mem S U
      V : Ultrafilter M
      hV : Membership.mem S V
      ⊢ Membership.mem S (HMul.hMul U V)
    -/
    rw [Set.mem_iInter] at *
    /-
      case refine_4
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      ⊢ ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => Memb …
    -/
    intro n
    /-
      case refine_4
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      n : Nat
      ⊢ Membership.mem (setOf fun U => Filter.Eventually (fun m => Membership.mem (H …
    -/
    rw [Set.mem_setOf_eq, Ultrafilter.eventually_mul]
    /-
      case refine_4
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      n : Nat
      ⊢ Filter.Eventually (fun m => Filter.Eventually (fun m' => Membership.mem (Hin …
    -/
    filter_upwards [hU n] with m hm
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      n : Nat
      m : M
      hm : Membership.mem (Hindman.FP (Stream'.drop n a)) m
      ⊢ Filter.Eventually (fun m' => Membership.mem (Hindman.FP (Stream'.drop n a))  …
    -/
    obtain ⟨n', hn⟩ := FP.mul hm
    /-
      case h.intro
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      n : Nat
      m : M
      hm : Membership.mem (Hindman.FP (Stream'.drop n a)) m
      n' : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n' (Stream'.drop n a …
      ⊢ Filter.Eventually (fun m' => Membership.mem (Hindman.FP (Stream'.drop n a))  …
    -/
    filter_upwards [hV (n' + n)] with m' hm'
    /-
      case h
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      n : Nat
      m : M
      hm : Membership.mem (Hindman.FP (Stream'.drop n a)) m
      n' : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n' (Stream'.drop n a …
      m' : M
      hm' : Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n' n) a)) m'
      ⊢ Membership.mem (Hindman.FP (Stream'.drop n a)) (HMul.hMul m m')
    -/
    apply hn
    /-
      case h.a
      M : Type u_1
      inst✝ : Semigroup M
      a : Stream' M
      S : Set (Ultrafilter M) := Set.iInter fun n => setOf fun U => Filter.Eventuall …
      U : Ultrafilter M
      hU : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      V : Ultrafilter M
      hV : ∀ (i : Nat), Membership.mem (setOf fun U => Filter.Eventually (fun m => M …
      n : Nat
      m : M
      hm : Membership.mem (Hindman.FP (Stream'.drop n a)) m
      n' : Nat
      hn : ∀ (m' : M), Membership.mem (Hindman.FP (Stream'.drop n' (Stream'.drop n a …
      m' : M
      hm' : Membership.mem (Hindman.FP (Stream'.drop (HAdd.hAdd n' n) a)) m'
      ⊢ Membership.mem (Hindman.FP (Stream'.drop n' (Stream'.drop n a))) m'
    -/
    simpa only [Stream'.drop_drop] using hm'
    /-
      🎉 no goals
    -/


@[to_additive exists_FS_of_large]
theorem exists_FP_of_large {M} [Semigroup M] (U : Ultrafilter M) (U_idem : U * U = U) (s₀ : Set M)
    (sU : s₀ ∈ U) : ∃ a, FP a ⊆ s₀ := by
  /- Informally: given a `U`-large set `s₀`, the set `s₀ ∩ { m | ∀ᶠ m' in U, m * m' ∈ s₀ }` is also
  `U`-large (since `U` is idempotent). Thus in particular there is an `a₀` in this intersection. Now
  let `s₁` be the intersection `s₀ ∩ { m | a₀ * m ∈ s₀ }`. By choice of `a₀`, this is again
  `U`-large, so we can repeat the argument starting from `s₁`, obtaining `a₁`, `s₂`, etc.
  This gives the desired infinite sequence. -/
  have exists_elem : ∀ {s : Set M} (_hs : s ∈ U), (s ∩ { m | ∀ᶠ m' in U, m * m' ∈ s }).Nonempty :=
    fun {s} hs => Ultrafilter.nonempty_of_mem (inter_mem hs <| by rwa [← U_idem] at hs)
  /-
    M : Type u_1
    inst✝ : Semigroup M
    U : Ultrafilter M
    U_idem : Eq (HMul.hMul U U) U
    s₀ : Set M
    sU : Membership.mem U s₀
    exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
    ⊢ Exists fun a => HasSubset.Subset (Hindman.FP a) s₀
  -/
  let elem : { s // s ∈ U } → M := fun p => (exists_elem p.property).some
  let succ : {s // s ∈ U} → {s // s ∈ U} := fun (p : {s // s ∈ U}) =>
        ⟨p.val ∩ {m : M | elem p * m ∈ p.val},
         inter_mem p.property
           (show (exists_elem p.property).some ∈ {m : M | ∀ᶠ (m' : M) in ↑U, m * m' ∈ p.val} from
              p.val.inter_subset_right (exists_elem p.property).some_mem)⟩
  /-
    M : Type u_1
    inst✝ : Semigroup M
    U : Ultrafilter M
    U_idem : Eq (HMul.hMul U U) U
    s₀ : Set M
    sU : Membership.mem U s₀
    exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
    elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
    succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
    ⊢ Exists fun a => HasSubset.Subset (Hindman.FP a) s₀
  -/
  use Stream'.corec elem succ (Subtype.mk s₀ sU)
  suffices ∀ (a : Stream' M), ∀ m ∈ FP a, ∀ p, a = Stream'.corec elem succ p → m ∈ p.val by
    intro m hm
    exact this _ m hm ⟨s₀, sU⟩ rfl
  /-
    case h
    M : Type u_1
    inst✝ : Semigroup M
    U : Ultrafilter M
    U_idem : Eq (HMul.hMul U U) U
    s₀ : Set M
    sU : Membership.mem U s₀
    exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
    elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
    succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
    ⊢ ∀ (a : Stream' M) (m : M), Membership.mem (Hindman.FP a) m → ∀ (p : Subtype  …
  -/
  clear sU s₀
  /-
    case h
    M : Type u_1
    inst✝ : Semigroup M
    U : Ultrafilter M
    U_idem : Eq (HMul.hMul U U) U
    exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
    elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
    succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
    ⊢ ∀ (a : Stream' M) (m : M), Membership.mem (Hindman.FP a) m → ∀ (p : Subtype  …
  -/
  intro a m h
  /-
    case h
    M : Type u_1
    inst✝ : Semigroup M
    U : Ultrafilter M
    U_idem : Eq (HMul.hMul U U) U
    exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
    elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
    succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
    a : Stream' M
    m : M
    h : Membership.mem (Hindman.FP a) m
    ⊢ ∀ (p : Subtype fun s => Membership.mem U s), Eq a (Stream'.corec elem succ p …
  -/
  induction' h with b b n h ih b n h ih
    /-
      case h.head
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m : M
      b : Stream' M
      ⊢ ∀ (p : Subtype fun s => Membership.mem U s), Eq b (Stream'.corec elem succ p …
    -/
  · rintro p rfl
    /-
      case h.head
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m : M
      p : Subtype fun s => Membership.mem U s
      ⊢ Membership.mem (↑p) (Stream'.corec elem succ p).head
    -/
    rw [Stream'.corec_eq, Stream'.head_cons]
    /-
      case h.head
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m : M
      p : Subtype fun s => Membership.mem U s
      ⊢ Membership.mem (↑p) (elem p)
    -/
    exact Set.inter_subset_left (Set.Nonempty.some_mem _)
    /-
      🎉 no goals
    -/
    /-
      case h.tail
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m : M
      b : Stream' M
      n : M
      h : Hindman.FP b.tail n
      ih : ∀ (p : Subtype fun s => Membership.mem U s), Eq b.tail (Stream'.corec ele …
      ⊢ ∀ (p : Subtype fun s => Membership.mem U s), Eq b (Stream'.corec elem succ p …
    -/
  · rintro p rfl
    /-
      case h.tail
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m n : M
      p : Subtype fun s => Membership.mem U s
      h : Hindman.FP (Stream'.corec elem succ p).tail n
      ih : ∀ (p_1 : Subtype fun s => Membership.mem U s), Eq (Stream'.corec elem suc …
      ⊢ Membership.mem (↑p) n
    -/
    refine Set.inter_subset_left (ih (succ p) ?_)
    /-
      case h.tail
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m n : M
      p : Subtype fun s => Membership.mem U s
      h : Hindman.FP (Stream'.corec elem succ p).tail n
      ih : ∀ (p_1 : Subtype fun s => Membership.mem U s), Eq (Stream'.corec elem suc …
      ⊢ Eq (Stream'.corec elem succ p).tail (Stream'.corec elem succ (succ p))
    -/
    rw [Stream'.corec_eq, Stream'.tail_cons]
    /-
      🎉 no goals
    -/
    /-
      case h.cons
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m : M
      b : Stream' M
      n : M
      h : Hindman.FP b.tail n
      ih : ∀ (p : Subtype fun s => Membership.mem U s), Eq b.tail (Stream'.corec ele …
      ⊢ ∀ (p : Subtype fun s => Membership.mem U s), Eq b (Stream'.corec elem succ p …
    -/
  · rintro p rfl
    /-
      case h.cons
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m n : M
      p : Subtype fun s => Membership.mem U s
      h : Hindman.FP (Stream'.corec elem succ p).tail n
      ih : ∀ (p_1 : Subtype fun s => Membership.mem U s), Eq (Stream'.corec elem suc …
      ⊢ Membership.mem (↑p) (HMul.hMul (Stream'.corec elem succ p).head n)
    -/
    have := Set.inter_subset_right (ih (succ p) ?_)
      /-
        case h.cons.refine_2
        M : Type u_1
        inst✝ : Semigroup M
        U : Ultrafilter M
        U_idem : Eq (HMul.hMul U U) U
        exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
        elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
        succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
        a : Stream' M
        m n : M
        p : Subtype fun s => Membership.mem U s
        h : Hindman.FP (Stream'.corec elem succ p).tail n
        ih : ∀ (p_1 : Subtype fun s => Membership.mem U s), Eq (Stream'.corec elem suc …
        this : Membership.mem (setOf fun m => Membership.mem (↑p) (HMul.hMul (elem p)  …
        ⊢ Membership.mem (↑p) (HMul.hMul (Stream'.corec elem succ p).head n)
      -/
    · simpa only using this
      /-
        🎉 no goals
      -/
    /-
      case h.cons.refine_1
      M : Type u_1
      inst✝ : Semigroup M
      U : Ultrafilter M
      U_idem : Eq (HMul.hMul U U) U
      exists_elem : ∀ {s : Set M}, Membership.mem U s → (Inter.inter s (setOf fun m  …
      elem : (Subtype fun s => Membership.mem U s) → M := fun p => ⋯.some
      succ : (Subtype fun s => Membership.mem U s) → Subtype fun s => Membership.mem …
      a : Stream' M
      m n : M
      p : Subtype fun s => Membership.mem U s
      h : Hindman.FP (Stream'.corec elem succ p).tail n
      ih : ∀ (p_1 : Subtype fun s => Membership.mem U s), Eq (Stream'.corec elem suc …
      ⊢ Eq (Stream'.corec elem succ p).tail (Stream'.corec elem succ (succ p))
    -/
    rw [Stream'.corec_eq, Stream'.tail_cons]
    /-
      🎉 no goals
    -/


/-- The strong form of **Hindman's theorem**: in any finite cover of an FP-set, one the parts
contains an FP-set. -/
@[to_additive FS_partition_regular
      "The strong form of **Hindman's theorem**: in any finite cover of
      an FS-set, one the parts contains an FS-set."]
theorem FP_partition_regular {M} [Semigroup M] (a : Stream' M) (s : Set (Set M)) (sfin : s.Finite)
    (scov : FP a ⊆ ⋃₀ s) : ∃ c ∈ s, ∃ b : Stream' M, FP b ⊆ c :=
  let ⟨U, idem, aU⟩ := exists_idempotent_ultrafilter_le_FP a
  let ⟨c, cs, hc⟩ := (Ultrafilter.finite_sUnion_mem_iff sfin).mp (mem_of_superset aU scov)
  ⟨c, cs, exists_FP_of_large U idem c hc⟩


/-- The weak form of **Hindman's theorem**: in any finite cover of a nonempty semigroup, one of the
parts contains an FP-set. -/
@[to_additive exists_FS_of_finite_cover
      "The weak form of **Hindman's theorem**: in any finite cover
      of a nonempty additive semigroup, one of the parts contains an FS-set."]
theorem exists_FP_of_finite_cover {M} [Semigroup M] [Nonempty M] (s : Set (Set M)) (sfin : s.Finite)
    (scov : ⊤ ⊆ ⋃₀ s) : ∃ c ∈ s, ∃ a : Stream' M, FP a ⊆ c :=
  let ⟨U, hU⟩ :=
    exists_idempotent_of_compact_t2_of_continuous_mul_left (@Ultrafilter.continuous_mul_left M _)
  let ⟨c, c_s, hc⟩ := (Ultrafilter.finite_sUnion_mem_iff sfin).mp (mem_of_superset univ_mem scov)
  ⟨c, c_s, exists_FP_of_large U hU c hc⟩


@[to_additive FS_iter_tail_sub_FS]
theorem FP_drop_subset_FP {M} [Semigroup M] (a : Stream' M) (n : ℕ) : FP (a.drop n) ⊆ FP a := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Nat.add_comm, ← Stream'.drop_drop]
    exact _root_.trans (FP.tail _) ih


@[to_additive]
theorem FP.singleton {M} [Semigroup M] (a : Stream' M) (i : ℕ) : a.get i ∈ FP a := by
  induction i generalizing a with
  | zero => exact FP.head _
  | succ i ih => exact FP.tail _ _ (ih _)


@[to_additive]
theorem FP.mul_two {M} [Semigroup M] (a : Stream' M) (i j : ℕ) (ij : i < j) :
    a.get i * a.get j ∈ FP a := by
  /-
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    ⊢ Membership.mem (Hindman.FP a) (HMul.hMul (a.get i) (a.get j))
  -/
  refine FP_drop_subset_FP _ i ?_
  /-
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    ⊢ Membership.mem (Hindman.FP (Stream'.drop i a)) (HMul.hMul (a.get i) (a.get j))
  -/
  rw [← Stream'.head_drop]
  /-
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    ⊢ Membership.mem (Hindman.FP (Stream'.drop i a)) (HMul.hMul (Stream'.drop i a) …
  -/
  apply FP.cons
  /-
    case h
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    ⊢ Hindman.FP (Stream'.drop i a).tail (a.get j)
  -/
  rcases Nat.exists_eq_add_of_le (Nat.succ_le_of_lt ij) with ⟨d, hd⟩
  -- Porting note: need to fix breakage of Set notation
  /-
    case h.intro
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    d : Nat
    hd : Eq j (HAdd.hAdd i.succ d)
    ⊢ Hindman.FP (Stream'.drop i a).tail (a.get j)
  -/
  change _ ∈ FP _
  /-
    case h.intro
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    d : Nat
    hd : Eq j (HAdd.hAdd i.succ d)
    ⊢ Membership.mem (Hindman.FP (Stream'.drop i a).tail) (a.get j)
  -/
  have := FP.singleton (a.drop i).tail d
  /-
    case h.intro
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    d : Nat
    hd : Eq j (HAdd.hAdd i.succ d)
    this : Membership.mem (Hindman.FP (Stream'.drop i a).tail) ((Stream'.drop i a) …
    ⊢ Membership.mem (Hindman.FP (Stream'.drop i a).tail) (a.get j)
  -/
  rw [Stream'.tail_eq_drop, Stream'.get_drop, Stream'.get_drop] at this
  /-
    case h.intro
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    d : Nat
    hd : Eq j (HAdd.hAdd i.succ d)
    this : Membership.mem (Hindman.FP (Stream'.drop 1 (Stream'.drop i a))) (a.get  …
    ⊢ Membership.mem (Hindman.FP (Stream'.drop i a).tail) (a.get j)
  -/
  convert this
  /-
    case h.e'_5.h.e'_3
    M : Type u_1
    inst✝ : Semigroup M
    a : Stream' M
    i j : Nat
    ij : LT.lt i j
    d : Nat
    hd : Eq j (HAdd.hAdd i.succ d)
    this : Membership.mem (Hindman.FP (Stream'.drop 1 (Stream'.drop i a))) (a.get  …
    ⊢ Eq j (HAdd.hAdd (HAdd.hAdd d 1) i)
  -/
  rw [hd, add_comm, Nat.succ_add, Nat.add_succ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem FP.finset_prod {M} [CommMonoid M] (a : Stream' M) (s : Finset ℕ) (hs : s.Nonempty) :
    (s.prod fun i => a.get i) ∈ FP a := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : Stream' M
    s : Finset Nat
    hs : s.Nonempty
    ⊢ Membership.mem (Hindman.FP a) (s.prod fun i => a.get i)
  -/
  refine FP_drop_subset_FP _ (s.min' hs) ?_
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : Stream' M
    s : Finset Nat
    hs : s.Nonempty
    ⊢ Membership.mem (Hindman.FP (Stream'.drop (s.min' hs) a)) (s.prod fun i => a. …
  -/
  induction' s using Finset.strongInduction with s ih
  /-
    case H
    M : Type u_1
    inst✝ : CommMonoid M
    a : Stream' M
    s : Finset Nat
    ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
    hs : s.Nonempty
    ⊢ Membership.mem (Hindman.FP (Stream'.drop (s.min' hs) a)) (s.prod fun i => a. …
  -/
  rw [← Finset.mul_prod_erase _ _ (s.min'_mem hs), ← Stream'.head_drop]
  /-
    case H
    M : Type u_1
    inst✝ : CommMonoid M
    a : Stream' M
    s : Finset Nat
    ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
    hs : s.Nonempty
    ⊢ Membership.mem (Hindman.FP (Stream'.drop (s.min' hs) a)) (HMul.hMul (Stream' …
  -/
  rcases (s.erase (s.min' hs)).eq_empty_or_nonempty with h | h
    /-
      case H.inl
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : Eq (s.erase (s.min' hs)) EmptyCollection.emptyCollection
      ⊢ Membership.mem (Hindman.FP (Stream'.drop (s.min' hs) a)) (HMul.hMul (Stream' …
    -/
  · rw [h, Finset.prod_empty, mul_one]
    /-
      case H.inl
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : Eq (s.erase (s.min' hs)) EmptyCollection.emptyCollection
      ⊢ Membership.mem (Hindman.FP (Stream'.drop (s.min' hs) a)) (Stream'.drop (s.mi …
    -/
    exact FP.head _
    /-
      🎉 no goals
    -/
    /-
      case H.inr
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : (s.erase (s.min' hs)).Nonempty
      ⊢ Membership.mem (Hindman.FP (Stream'.drop (s.min' hs) a)) (HMul.hMul (Stream' …
    -/
  · apply FP.cons
    /-
      case H.inr.h
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : (s.erase (s.min' hs)).Nonempty
      ⊢ Hindman.FP (Stream'.drop (s.min' hs) a).tail ((s.erase (s.min' hs)).prod fun …
    -/
    rw [Stream'.tail_eq_drop, Stream'.drop_drop, add_comm]
    /-
      case H.inr.h
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : (s.erase (s.min' hs)).Nonempty
      ⊢ Hindman.FP (Stream'.drop (HAdd.hAdd (s.min' hs) 1) a) ((s.erase (s.min' hs)) …
    -/
    refine Set.mem_of_subset_of_mem ?_ (ih _ (Finset.erase_ssubset <| s.min'_mem hs) h)
    have : s.min' hs + 1 ≤ (s.erase (s.min' hs)).min' h :=
      Nat.succ_le_of_lt (Finset.min'_lt_of_mem_erase_min' _ _ <| Finset.min'_mem _ _)
    /-
      case H.inr.h
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : (s.erase (s.min' hs)).Nonempty
      this : LE.le (HAdd.hAdd (s.min' hs) 1) ((s.erase (s.min' hs)).min' h)
      ⊢ HasSubset.Subset (Hindman.FP (Stream'.drop ((s.erase (s.min' hs)).min' h) a) …
    -/
    cases' Nat.exists_eq_add_of_le this with d hd
    /-
      case H.inr.h.intro
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : (s.erase (s.min' hs)).Nonempty
      this : LE.le (HAdd.hAdd (s.min' hs) 1) ((s.erase (s.min' hs)).min' h)
      d : Nat
      hd : Eq ((s.erase (s.min' hs)).min' h) (HAdd.hAdd (HAdd.hAdd (s.min' hs) 1) d)
      ⊢ HasSubset.Subset (Hindman.FP (Stream'.drop ((s.erase (s.min' hs)).min' h) a) …
    -/
    rw [hd, add_comm, ← Stream'.drop_drop]
    /-
      case H.inr.h.intro
      M : Type u_1
      inst✝ : CommMonoid M
      a : Stream' M
      s : Finset Nat
      ih : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (hs : t.Nonempty), Members …
      hs : s.Nonempty
      h : (s.erase (s.min' hs)).Nonempty
      this : LE.le (HAdd.hAdd (s.min' hs) 1) ((s.erase (s.min' hs)).min' h)
      d : Nat
      hd : Eq ((s.erase (s.min' hs)).min' h) (HAdd.hAdd (HAdd.hAdd (s.min' hs) 1) d)
      ⊢ HasSubset.Subset (Hindman.FP (Stream'.drop d (Stream'.drop (HAdd.hAdd (s.min …
    -/
    apply FP_drop_subset_FP
    /-
      🎉 no goals
    -/


