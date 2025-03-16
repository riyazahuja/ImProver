/-- Encodes a term as a list of variables and function symbols. -/
def listEncode : L.Term α → List (α ⊕ (Σi, L.Functions i))
  | var i => [Sum.inl i]
  | func f ts =>
    Sum.inr (⟨_, f⟩ : Σi, L.Functions i)::(List.finRange _).flatMap fun i => (ts i).listEncode


/-- Decodes a list of variables and function symbols as a list of terms. -/
def listDecode : List (α ⊕ (Σi, L.Functions i)) → List (L.Term α)
  | [] => []
  | Sum.inl a::l => (var a)::listDecode l
  | Sum.inr ⟨n, f⟩::l =>
    if h : n ≤ (listDecode l).length then
      (func f (fun i => (listDecode l)[i])) :: (listDecode l).drop n
    else []


theorem listDecode_encode_list (l : List (L.Term α)) :
    listDecode (l.flatMap listEncode) = l := by
  suffices h : ∀ (t : L.Term α) (l : List (α ⊕ (Σi, L.Functions i))),
      listDecode (t.listEncode ++ l) = t::listDecode l by
    induction' l with t l lih
    · rfl
    · rw [flatMap_cons, h t (l.flatMap listEncode), lih]
  /-
    L : FirstOrder.Language
    α : Type u'
    l : List (L.Term α)
    ⊢ ∀ (t : L.Term α) (l : List (Sum α (Sigma fun i => L.Functions i))), Eq (Firs …
  -/
  intro t
  /-
    L : FirstOrder.Language
    α : Type u'
    l : List (L.Term α)
    t : L.Term α
    ⊢ ∀ (l : List (Sum α (Sigma fun i => L.Functions i))), Eq (FirstOrder.Language …
  -/
  induction' t with a n f ts ih <;> intro l
    /-
      case var
      L : FirstOrder.Language
      α : Type u'
      l✝ : List (L.Term α)
      a : α
      l : List (Sum α (Sigma fun i => L.Functions i))
      ⊢ Eq (FirstOrder.Language.Term.listDecode (HAppend.hAppend (FirstOrder.Languag …
    -/
  · rw [listEncode, singleton_append, listDecode]
    /-
      🎉 no goals
    -/
    /-
      case func
      L : FirstOrder.Language
      α : Type u'
      l✝ : List (L.Term α)
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term α
      ih : ∀ (a : Fin n) (l : List (Sum α (Sigma fun i => L.Functions i))), Eq (Firs …
      l : List (Sum α (Sigma fun i => L.Functions i))
      ⊢ Eq (FirstOrder.Language.Term.listDecode (HAppend.hAppend (FirstOrder.Languag …
    -/
  · rw [listEncode, cons_append, listDecode]
    have h : listDecode (((finRange n).flatMap fun i : Fin n => (ts i).listEncode) ++ l) =
        (finRange n).map ts ++ listDecode l := by
      induction' finRange n with i l' l'ih
      · rfl
      · rw [flatMap_cons, List.append_assoc, ih, map_cons, l'ih, cons_append]
    simp only [h, length_append, length_map, length_finRange, le_add_iff_nonneg_right,
      _root_.zero_le, ↓reduceDIte, getElem_fin, cons.injEq, func.injEq, heq_eq_eq, true_and]
    /-
      case func
      L : FirstOrder.Language
      α : Type u'
      l✝ : List (L.Term α)
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term α
      ih : ∀ (a : Fin n) (l : List (Sum α (Sigma fun i => L.Functions i))), Eq (Firs …
      l : List (Sum α (Sigma fun i => L.Functions i))
      h : Eq (FirstOrder.Language.Term.listDecode (HAppend.hAppend ((List.finRange n …
      ⊢ And (Eq (fun i => GetElem.getElem (HAppend.hAppend (List.map ts (List.finRan …
    -/
    refine ⟨funext (fun i => ?_), ?_⟩
    · simp only [length_map, length_finRange, is_lt, getElem_append_left, getElem_map,
      getElem_finRange, cast_mk, Fin.eta]
      /-
        case func.refine_2
        L : FirstOrder.Language
        α : Type u'
        l✝ : List (L.Term α)
        n : Nat
        f : L.Functions n
        ts : Fin n → L.Term α
        ih : ∀ (a : Fin n) (l : List (Sum α (Sigma fun i => L.Functions i))), Eq (Firs …
        l : List (Sum α (Sigma fun i => L.Functions i))
        h : Eq (FirstOrder.Language.Term.listDecode (HAppend.hAppend ((List.finRange n …
        ⊢ Eq (List.drop n (HAppend.hAppend (List.map ts (List.finRange n)) (FirstOrder …
      -/
    · simp only [length_map, length_finRange, drop_left']
      /-
        🎉 no goals
      -/


/-- An encoding of terms as lists. -/
@[simps]
protected def encoding : Encoding (L.Term α) where
  Γ := α ⊕ (Σi, L.Functions i)
  encode := listEncode
  decode l := (listDecode l).head?.join
  decode_encode t := by
    /-
      L : FirstOrder.Language
      α : Type u'
      t : L.Term α
      ⊢ Eq ((fun l => (Bind.bind (FirstOrder.Language.Term.listDecode l).head? fun a …
    -/
    have h := listDecode_encode_list [t]
    /-
      L : FirstOrder.Language
      α : Type u'
      t : L.Term α
      h : Eq (FirstOrder.Language.Term.listDecode ((List.cons t List.nil).flatMap Fi …
      ⊢ Eq ((fun l => (Bind.bind (FirstOrder.Language.Term.listDecode l).head? fun a …
    -/
    rw [flatMap_singleton] at h
    simp only [Option.join, h, head?_cons, Option.pure_def, Option.bind_eq_bind, Option.some_bind,
      id_eq]


theorem listEncode_injective :
    Function.Injective (listEncode : L.Term α → List (α ⊕ (Σi, L.Functions i))) :=
  Term.encoding.encode_injective


theorem card_le : #(L.Term α) ≤ max ℵ₀ #(α ⊕ (Σi, L.Functions i)) :=
  lift_le.1 (_root_.trans Term.encoding.card_le_card_list (lift_le.2 (mk_list_le_max _)))


theorem card_sigma : #(Σn, L.Term (α ⊕ (Fin n))) = max ℵ₀ #(α ⊕ (Σi, L.Functions i)) := by
  /-
    L : FirstOrder.Language
    α : Type u'
    ⊢ Eq (Cardinal.mk (Sigma fun n => L.Term (Sum α (Fin n)))) (Max.max Cardinal.a …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      L : FirstOrder.Language
      α : Type u'
      ⊢ LE.le (Cardinal.mk (Sigma fun n => L.Term (Sum α (Fin n)))) (Max.max Cardina …
    -/
  · rw [mk_sigma]
    /-
      case refine_1
      L : FirstOrder.Language
      α : Type u'
      ⊢ LE.le (Cardinal.sum fun i => Cardinal.mk (L.Term (Sum α (Fin i)))) (Max.max  …
    -/
    refine (sum_le_iSup_lift _).trans ?_
    rw [mk_nat, lift_aleph0, mul_eq_max_of_aleph0_le_left le_rfl, max_le_iff,
      ciSup_le_iff' (bddAbove_range _)]
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        ⊢ And (LE.le Cardinal.aleph0 (Max.max Cardinal.aleph0 (Cardinal.mk (Sum α (Sig …
      -/
    · refine ⟨le_max_left _ _, fun i => card_le.trans ?_⟩
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        i : Nat
        ⊢ LE.le (Max.max Cardinal.aleph0 (Cardinal.mk (Sum (Sum α (Fin i)) (Sigma fun  …
      -/
      refine max_le (le_max_left _ _) ?_
      rw [← add_eq_max le_rfl, mk_sum, mk_sum, mk_sum, add_comm (Cardinal.lift #α), lift_add,
        add_assoc, lift_lift, lift_lift, mk_fin, lift_natCast]
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        i : Nat
        ⊢ LE.le (HAdd.hAdd (↑i) (HAdd.hAdd (Cardinal.lift.{u, u'} (Cardinal.mk α)) (Ca …
      -/
      exact add_le_add_right (nat_lt_aleph0 _).le _
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        ⊢ Ne (iSup fun i => Cardinal.mk (L.Term (Sum α (Fin i)))) 0
      -/
    · rw [← one_le_iff_ne_zero]
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        ⊢ LE.le 1 (iSup fun i => Cardinal.mk (L.Term (Sum α (Fin i))))
      -/
      refine _root_.trans ?_ (le_ciSup (bddAbove_range _) 1)
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        ⊢ LE.le 1 (Cardinal.mk (L.Term (Sum α (Fin 1))))
      -/
      rw [one_le_iff_ne_zero, mk_ne_zero_iff]
      /-
        case refine_1
        L : FirstOrder.Language
        α : Type u'
        ⊢ Nonempty (L.Term (Sum α (Fin 1)))
      -/
      exact ⟨var (Sum.inr 0)⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      L : FirstOrder.Language
      α : Type u'
      ⊢ LE.le (Max.max Cardinal.aleph0 (Cardinal.mk (Sum α (Sigma fun i => L.Functio …
    -/
  · rw [max_le_iff, ← infinite_iff]
    /-
      case refine_2
      L : FirstOrder.Language
      α : Type u'
      ⊢ And (Infinite (Sigma fun n => L.Term (Sum α (Fin n)))) (LE.le (Cardinal.mk ( …
    -/
    refine ⟨Infinite.of_injective (fun i => ⟨i + 1, var (Sum.inr i)⟩) fun i j ij => ?_, ?_⟩
      /-
        case refine_2.refine_1
        L : FirstOrder.Language
        α : Type u'
        i j : Nat
        ij : Eq ((fun i => ⟨HAdd.hAdd i 1, FirstOrder.Language.Term.var (Sum.inr ↑i)⟩) …
        ⊢ Eq i j
      -/
    · cases ij
      /-
        case refine_2.refine_1.refl
        L : FirstOrder.Language
        α : Type u'
        i : Nat
        ⊢ Eq i i
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        L : FirstOrder.Language
        α : Type u'
        ⊢ LE.le (Cardinal.mk (Sum α (Sigma fun i => L.Functions i))) (Cardinal.mk (Sig …
      -/
    · rw [Cardinal.le_def]
      refine ⟨⟨Sum.elim (fun i => ⟨0, var (Sum.inl i)⟩)
        fun F => ⟨1, func F.2 fun _ => var (Sum.inr 0)⟩, ?_⟩⟩
      /-
        case refine_2.refine_2
        L : FirstOrder.Language
        α : Type u'
        ⊢ Function.Injective (Sum.elim (fun i => ⟨0, FirstOrder.Language.Term.var (Sum …
      -/
      rintro (a | a) (b | b) h
      · simp only [Sum.elim_inl, Sigma.mk.inj_iff, heq_eq_eq, var.injEq, Sum.inl.injEq, true_and]
          at h
        /-
          case refine_2.refine_2.inl.inl
          L : FirstOrder.Language
          α : Type u'
          a b : α
          h : Eq a b
          ⊢ Eq (Sum.inl a) (Sum.inl b)
        -/
        rw [h]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2.inl.inr
          L : FirstOrder.Language
          α : Type u'
          a : α
          b : Sigma fun i => L.Functions i
          h : Eq (Sum.elim (fun i => ⟨0, FirstOrder.Language.Term.var (Sum.inl i)⟩) (fun …
          ⊢ Eq (Sum.inl a) (Sum.inr b)
        -/
      · simp only [Sum.elim_inl, Sum.elim_inr, Sigma.mk.inj_iff, false_and, reduceCtorEq] at h
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2.inr.inl
          L : FirstOrder.Language
          α : Type u'
          a : Sigma fun i => L.Functions i
          b : α
          h : Eq (Sum.elim (fun i => ⟨0, FirstOrder.Language.Term.var (Sum.inl i)⟩) (fun …
          ⊢ Eq (Sum.inr a) (Sum.inl b)
        -/
      · simp only [Sum.elim_inr, Sum.elim_inl, Sigma.mk.inj_iff, false_and, reduceCtorEq] at h
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2.inr.inr
          L : FirstOrder.Language
          α : Type u'
          a b : Sigma fun i => L.Functions i
          h : Eq (Sum.elim (fun i => ⟨0, FirstOrder.Language.Term.var (Sum.inl i)⟩) (fun …
          ⊢ Eq (Sum.inr a) (Sum.inr b)
        -/
      · simp only [Sum.elim_inr, Sigma.mk.inj_iff, heq_eq_eq, func.injEq, true_and] at h
        /-
          case refine_2.refine_2.inr.inr
          L : FirstOrder.Language
          α : Type u'
          a b : Sigma fun i => L.Functions i
          h : And (Eq a.fst b.fst) (And (HEq a.snd b.snd) (HEq (fun x => FirstOrder.Lang …
          ⊢ Eq (Sum.inr a) (Sum.inr b)
        -/
        rw [Sigma.ext_iff.2 ⟨h.1, h.2.1⟩]
        /-
          🎉 no goals
        -/


instance [Encodable α] [Encodable (Σi, L.Functions i)] : Encodable (L.Term α) :=
  Encodable.ofLeftInjection listEncode (fun l => (listDecode l).head?.join) fun t => by
    /-
      L : FirstOrder.Language
      α : Type u'
      inst✝¹ : Encodable α
      inst✝ : Encodable (Sigma fun i => L.Functions i)
      t : L.Term α
      ⊢ Eq ((fun l => (Bind.bind (FirstOrder.Language.Term.listDecode l).head? fun a …
    -/
    simp only
    /-
      L : FirstOrder.Language
      α : Type u'
      inst✝¹ : Encodable α
      inst✝ : Encodable (Sigma fun i => L.Functions i)
      t : L.Term α
      ⊢ Eq (Bind.bind (FirstOrder.Language.Term.listDecode t.listEncode).head? fun a …
    -/
    rw [← flatMap_singleton listEncode, listDecode_encode_list]
    simp only [Option.join, head?_cons, Option.pure_def, Option.bind_eq_bind, Option.some_bind,
      id_eq]


instance [h1 : Countable α] [h2 : Countable (Σl, L.Functions l)] : Countable (L.Term α) := by
  /-
    L : FirstOrder.Language
    α : Type u'
    h1 : Countable α
    h2 : Countable (Sigma fun l => L.Functions l)
    ⊢ Countable (L.Term α)
  -/
  refine mk_le_aleph0_iff.1 (card_le.trans (max_le_iff.2 ?_))
  /-
    L : FirstOrder.Language
    α : Type u'
    h1 : Countable α
    h2 : Countable (Sigma fun l => L.Functions l)
    ⊢ And (LE.le Cardinal.aleph0 Cardinal.aleph0) (LE.le (Cardinal.mk (Sum α (Sigm …
  -/
  simp only [le_refl, mk_sum, add_le_aleph0, lift_le_aleph0, true_and]
  /-
    L : FirstOrder.Language
    α : Type u'
    h1 : Countable α
    h2 : Countable (Sigma fun l => L.Functions l)
    ⊢ And (LE.le (Cardinal.mk α) Cardinal.aleph0) (LE.le (Cardinal.mk (Sigma fun i …
  -/
  exact ⟨Cardinal.mk_le_aleph0, Cardinal.mk_le_aleph0⟩
  /-
    🎉 no goals
  -/


instance small [Small.{u} α] : Small.{u} (L.Term α) :=
  small_of_injective listEncode_injective


/-- Encodes a bounded formula as a list of symbols. -/
def listEncode : ∀ {n : ℕ},
    L.BoundedFormula α n → List ((Σk, L.Term (α ⊕ Fin k)) ⊕ ((Σn, L.Relations n) ⊕ ℕ))
  | n, falsum => [Sum.inr (Sum.inr (n + 2))]
  | _, equal t₁ t₂ => [Sum.inl ⟨_, t₁⟩, Sum.inl ⟨_, t₂⟩]
  | n, rel R ts => [Sum.inr (Sum.inl ⟨_, R⟩), Sum.inr (Sum.inr n)] ++
      (List.finRange _).map fun i => Sum.inl ⟨n, ts i⟩
  | _, imp φ₁ φ₂ => (Sum.inr (Sum.inr 0)::φ₁.listEncode) ++ φ₂.listEncode
  | _, all φ => Sum.inr (Sum.inr 1)::φ.listEncode


/-- Applies the `forall` quantifier to an element of `(Σ n, L.BoundedFormula α n)`,
or returns `default` if not possible. -/
def sigmaAll : (Σn, L.BoundedFormula α n) → Σn, L.BoundedFormula α n
  | ⟨n + 1, φ⟩ => ⟨n, φ.all⟩
  | _ => default



@[simp]
lemma sigmaAll_apply {n} {φ : L.BoundedFormula α (n + 1)} :
    sigmaAll ⟨n + 1, φ⟩ = ⟨n, φ.all⟩ := rfl


/-- Applies `imp` to two elements of `(Σ n, L.BoundedFormula α n)`,
or returns `default` if not possible. -/
def sigmaImp : (Σn, L.BoundedFormula α n) → (Σn, L.BoundedFormula α n) → Σn, L.BoundedFormula α n
                                                             /-
                                                               L : FirstOrder.Language
                                                               α : Type u'
                                                               m : Nat
                                                               φ : L.BoundedFormula α m
                                                               n : Nat
                                                               ψ : L.BoundedFormula α n
                                                               h : Eq m n
                                                               ⊢ Eq (L.BoundedFormula α n) (L.BoundedFormula α m)
                                                             -/
  | ⟨m, φ⟩, ⟨n, ψ⟩ => if h : m = n then ⟨m, φ.imp (Eq.mp (by rw [h]) ψ)⟩ else default
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Decodes a list of symbols as a list of formulas. -/
@[simp]
lemma sigmaImp_apply {n} {φ ψ : L.BoundedFormula α n} :
    sigmaImp ⟨n, φ⟩ ⟨n, ψ⟩ = ⟨n, φ.imp ψ⟩ := by
  /-
    L : FirstOrder.Language
    α : Type u'
    n : Nat
    φ ψ : L.BoundedFormula α n
    ⊢ Eq (FirstOrder.Language.BoundedFormula.sigmaImp ⟨n, φ⟩ ⟨n, ψ⟩) ⟨n, φ.imp ψ⟩
  -/
  simp only [sigmaImp, ↓reduceDIte, eq_mp_eq_cast, cast_eq]
  /-
    🎉 no goals
  -/


/-- Decodes a list of symbols as a list of formulas. -/
def listDecode :
    List ((Σk, L.Term (α ⊕ Fin k)) ⊕ ((Σn, L.Relations n) ⊕ ℕ)) → List (Σn, L.BoundedFormula α n)
  | Sum.inr (Sum.inr (n + 2))::l => ⟨n, falsum⟩::(listDecode l)
  | Sum.inl ⟨n₁, t₁⟩::Sum.inl ⟨n₂, t₂⟩::l =>
                                                  /-
                                                    L : FirstOrder.Language
                                                    α : Type u'
                                                    n₁ : Nat
                                                    t₁ : L.Term (Sum α (Fin n₁))
                                                    n₂ : Nat
                                                    t₂ : L.Term (Sum α (Fin n₂))
                                                    l : List (Sum (Sigma fun k => L.Term (Sum α (Fin k))) (Sum (Sigma fun n => L.R …
                                                    h : Eq n₁ n₂
                                                    ⊢ Eq (L.Term (Sum α (Fin n₂))) (L.Term (Sum α (Fin n₁)))
                                                  -/
    (if h : n₁ = n₂ then ⟨n₁, equal t₁ (Eq.mp (by rw [h]) t₂)⟩ else default)::(listDecode l)
                                                  /-
                                                    🎉 no goals
                                                  -/
  | Sum.inr (Sum.inl ⟨n, R⟩)::Sum.inr (Sum.inr k)::l => (
    if h : ∀ i : Fin n, ((l.map Sum.getLeft?).get? i).join.isSome then
        if h' : ∀ i, (Option.get _ (h i)).1 = k then
                                                      /-
                                                        L : FirstOrder.Language
                                                        α : Type u'
                                                        n : Nat
                                                        R : L.Relations n
                                                        k : Nat
                                                        l : List (Sum (Sigma fun k => L.Term (Sum α (Fin k))) (Sum (Sigma fun n => L.R …
                                                        h : ∀ (i : Fin n), Eq ((List.map Sum.getLeft? l).get? ↑i).join.isSome Bool.true
                                                        h' : ∀ (i : Fin n), Eq (((List.map Sum.getLeft? l).get? ↑i).join.get ⋯).fst k
                                                        i : Fin n
                                                        ⊢ Eq (L.Term (Sum α (Fin (((List.map Sum.getLeft? l).get? ↑i).join.get ⋯).fst) …
                                                      -/
          ⟨k, BoundedFormula.rel R fun i => Eq.mp (by rw [h' i]) (Option.get _ (h i)).2⟩
                                                      /-
                                                        🎉 no goals
                                                      -/
        else default
      else default)::(listDecode (l.drop n))
  | Sum.inr (Sum.inr 0)::l => if h : 2 ≤ (listDecode l).length
    then (sigmaImp (listDecode l)[0] (listDecode l)[1])::(drop 2 (listDecode l))
    else []
  | Sum.inr (Sum.inr 1)::l => if h : 1 ≤ (listDecode l).length
    then (sigmaAll (listDecode l)[0])::(drop 1 (listDecode l))
    else []
  | _ => []
  termination_by l => l.length


@[simp]
theorem listDecode_encode_list (l : List (Σn, L.BoundedFormula α n)) :
    listDecode (l.flatMap (fun φ => φ.2.listEncode)) = l := by
  suffices h : ∀ (φ : Σn, L.BoundedFormula α n)
      (l' : List ((Σk, L.Term (α ⊕ Fin k)) ⊕ ((Σn, L.Relations n) ⊕ ℕ))),
      (listDecode (listEncode φ.2 ++ l')) = φ::(listDecode l') by
    induction' l with φ l ih
    · rw [List.flatMap_nil]
      simp [listDecode]
    · rw [flatMap_cons, h φ _, ih]
  /-
    L : FirstOrder.Language
    α : Type u'
    l : List (Sigma fun n => L.BoundedFormula α n)
    ⊢ ∀ (φ : Sigma fun n => L.BoundedFormula α n) (l' : List (Sum (Sigma fun k =>  …
  -/
  rintro ⟨n, φ⟩
  induction φ with
  | falsum => intro l; rw [listEncode, singleton_append, listDecode]
  | equal =>
    intro l
    rw [listEncode, cons_append, cons_append, listDecode, dif_pos]
    · simp only [eq_mp_eq_cast, cast_eq, eq_self_iff_true, heq_iff_eq, and_self_iff, nil_append]
    · simp only [eq_self_iff_true, heq_iff_eq, and_self_iff]
  | @rel φ_n φ_l φ_R ts =>
    intro l
    rw [listEncode, cons_append, cons_append, singleton_append, cons_append, listDecode]
    have h : ∀ i : Fin φ_l, ((List.map Sum.getLeft? (List.map (fun i : Fin φ_l =>
      Sum.inl (⟨(⟨φ_n, rel φ_R ts⟩ : Σn, L.BoundedFormula α n).fst, ts i⟩ :
        Σn, L.Term (α ⊕ (Fin n)))) (finRange φ_l) ++ l)).get? ↑i).join = some ⟨_, ts i⟩ := by
      intro i
      simp only [Option.join, map_append, map_map, Option.bind_eq_some, id, exists_eq_right,
        get?_eq_some_iff, length_append, length_map, length_finRange]
      refine ⟨lt_of_lt_of_le i.2 le_self_add, ?_⟩
      rw [get_eq_getElem, getElem_append_left, getElem_map]
      · simp only [getElem_finRange, cast_mk, Fin.eta, Function.comp_apply, Sum.getLeft?_inl]
      · simp only [length_map, length_finRange, is_lt]
    rw [dif_pos]
    swap
    · exact fun i => Option.isSome_iff_exists.2 ⟨⟨_, ts i⟩, h i⟩
    rw [dif_pos]
    swap
    · intro i
      obtain ⟨h1, h2⟩ := Option.eq_some_iff_get_eq.1 (h i)
      rw [h2]
    simp only [Option.join, eq_mp_eq_cast, cons.injEq, Sigma.mk.inj_iff, heq_eq_eq, rel.injEq,
      true_and]
    refine ⟨funext fun i => ?_, ?_⟩
    · obtain ⟨h1, h2⟩ := Option.eq_some_iff_get_eq.1 (h i)
      rw [cast_eq_iff_heq]
      exact (Sigma.ext_iff.1 ((Sigma.eta (Option.get _ h1)).trans h2)).2
    rw [List.drop_append_eq_append_drop, length_map, length_finRange, Nat.sub_self, drop,
      drop_eq_nil_of_le, nil_append]
    rw [length_map, length_finRange]
  | imp _ _ ih1 ih2 =>
    intro l
    simp only [] at *
    rw [listEncode, List.append_assoc, cons_append, listDecode]
    simp only [ih1, ih2, length_cons, le_add_iff_nonneg_left, _root_.zero_le, ↓reduceDIte,
      getElem_cons_zero, getElem_cons_succ, sigmaImp_apply, drop_succ_cons, drop_zero]
  | all _ ih =>
    intro l
    simp only [] at *
    rw [listEncode, cons_append, listDecode]
    simp only [ih, length_cons, le_add_iff_nonneg_left, _root_.zero_le, ↓reduceDIte,
      getElem_cons_zero, sigmaAll_apply, drop_succ_cons, drop_zero]


/-- An encoding of bounded formulas as lists. -/
@[simps]
protected def encoding : Encoding (Σn, L.BoundedFormula α n) where
  Γ := (Σk, L.Term (α ⊕ Fin k)) ⊕ ((Σn, L.Relations n) ⊕ ℕ)
  encode φ := φ.2.listEncode
  decode l := (listDecode l)[0]?
  decode_encode φ := by
    /-
      L : FirstOrder.Language
      α : Type u'
      φ : Sigma fun n => L.BoundedFormula α n
      ⊢ Eq ((fun l => GetElem?.getElem? (FirstOrder.Language.BoundedFormula.listDeco …
    -/
    have h := listDecode_encode_list [φ]
    /-
      L : FirstOrder.Language
      α : Type u'
      φ : Sigma fun n => L.BoundedFormula α n
      h : Eq (FirstOrder.Language.BoundedFormula.listDecode ((List.cons φ List.nil). …
      ⊢ Eq ((fun l => GetElem?.getElem? (FirstOrder.Language.BoundedFormula.listDeco …
    -/
    rw [flatMap_singleton] at h
    /-
      L : FirstOrder.Language
      α : Type u'
      φ : Sigma fun n => L.BoundedFormula α n
      h : Eq (FirstOrder.Language.BoundedFormula.listDecode φ.snd.listEncode) (List. …
      ⊢ Eq ((fun l => GetElem?.getElem? (FirstOrder.Language.BoundedFormula.listDeco …
    -/
    simp only
    /-
      L : FirstOrder.Language
      α : Type u'
      φ : Sigma fun n => L.BoundedFormula α n
      h : Eq (FirstOrder.Language.BoundedFormula.listDecode φ.snd.listEncode) (List. …
      ⊢ Eq (GetElem?.getElem? (FirstOrder.Language.BoundedFormula.listDecode φ.snd.l …
    -/
    rw [h]
    /-
      L : FirstOrder.Language
      α : Type u'
      φ : Sigma fun n => L.BoundedFormula α n
      h : Eq (FirstOrder.Language.BoundedFormula.listDecode φ.snd.listEncode) (List. …
      ⊢ Eq (GetElem?.getElem? (List.cons φ List.nil) 0) (Option.some φ)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem listEncode_sigma_injective :
    Function.Injective fun φ : Σn, L.BoundedFormula α n => φ.2.listEncode :=
  BoundedFormula.encoding.encode_injective


theorem card_le : #(Σn, L.BoundedFormula α n) ≤
    max ℵ₀ (Cardinal.lift.{max u v} #α + Cardinal.lift.{u'} L.card) := by
  /-
    L : FirstOrder.Language
    α : Type u'
    ⊢ LE.le (Cardinal.mk (Sigma fun n => L.BoundedFormula α n)) (Max.max Cardinal. …
  -/
  refine lift_le.1 (BoundedFormula.encoding.card_le_card_list.trans ?_)
  rw [encoding_Γ, mk_list_eq_max_mk_aleph0, lift_max, lift_aleph0, lift_max, lift_aleph0,
    max_le_iff]
  /-
    L : FirstOrder.Language
    α : Type u'
    ⊢ And (LE.le (Cardinal.lift.{max (max u u') v, max (max u u') v} (Cardinal.mk  …
  -/
  refine ⟨?_, le_max_left _ _⟩
  /-
    L : FirstOrder.Language
    α : Type u'
    ⊢ LE.le (Cardinal.lift.{max (max u u') v, max (max u u') v} (Cardinal.mk (Sum  …
  -/
  rw [mk_sum, Term.card_sigma, mk_sum, ← add_eq_max le_rfl, mk_sum, mk_nat]
  /-
    L : FirstOrder.Language
    α : Type u'
    ⊢ LE.le (Cardinal.lift.{max (max u u') v, max (max u u') v} (HAdd.hAdd (Cardin …
  -/
  simp only [lift_add, lift_lift, lift_aleph0]
  rw [← add_assoc, add_comm, ← add_assoc, ← add_assoc, aleph0_add_aleph0, add_assoc,
    add_eq_max le_rfl, add_assoc, card, Symbols, mk_sum, lift_add, lift_lift, lift_lift]


