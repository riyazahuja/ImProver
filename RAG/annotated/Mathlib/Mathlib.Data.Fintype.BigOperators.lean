@[to_additive]
                                                                                    /-
                                                                                      α : Type u_1
                                                                                      inst✝ : CommMonoid α
                                                                                      f : Bool → α
                                                                                      ⊢ Eq (Finset.univ.prod fun b => f b) (HMul.hMul (f Bool.true) (f Bool.false))
                                                                                    -/
theorem prod_bool [CommMonoid α] (f : Bool → α) : ∏ b, f b = f true * f false := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem card_eq_sum_ones {α} [Fintype α] : Fintype.card α = ∑ _a : α, 1 :=
  Finset.card_eq_sum_ones _


@[to_additive]
theorem prod_extend_by_one [CommMonoid α] (s : Finset ι) (f : ι → α) :
    ∏ i, (if i ∈ s then f i else 1) = ∏ i ∈ s, f i := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : CommMonoid α
    s : Finset ι
    f : ι → α
    ⊢ Eq (Finset.univ.prod fun i => ite (Membership.mem s i) (f i) 1) (s.prod fun  …
  -/
  rw [← prod_filter, filter_mem_eq_inter, univ_inter]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_eq_one (f : α → M) (h : ∀ a, f a = 1) : ∏ a, f a = 1 :=
  Finset.prod_eq_one fun a _ha => h a


@[to_additive]
theorem prod_congr (f g : α → M) (h : ∀ a, f a = g a) : ∏ a, f a = ∏ a, g a :=
  Finset.prod_congr rfl fun a _ha => h a


@[to_additive]
theorem prod_eq_single {f : α → M} (a : α) (h : ∀ x ≠ a, f x = 1) : ∏ x, f x = f a :=
  Finset.prod_eq_single a (fun x _ hx => h x hx) fun ha => (ha (Finset.mem_univ a)).elim


@[to_additive]
theorem prod_eq_mul {f : α → M} (a b : α) (h₁ : a ≠ b) (h₂ : ∀ x, x ≠ a ∧ x ≠ b → f x = 1) :
    ∏ x, f x = f a * f b := by
  /-
    α : Type u_1
    M : Type u_4
    inst✝¹ : Fintype α
    inst✝ : CommMonoid M
    f : α → M
    a b : α
    h₁ : Ne a b
    h₂ : ∀ (x : α), And (Ne x a) (Ne x b) → Eq (f x) 1
    ⊢ Eq (Finset.univ.prod fun x => f x) (HMul.hMul (f a) (f b))
  -/
  apply Finset.prod_eq_mul a b h₁ fun x _ hx => h₂ x hx <;>
    /-
      case ha
      α : Type u_1
      M : Type u_4
      inst✝¹ : Fintype α
      inst✝ : CommMonoid M
      f : α → M
      a b : α
      h₁ : Ne a b
      h₂ : ∀ (x : α), And (Ne x a) (Ne x b) → Eq (f x) 1
      ⊢ Not (Membership.mem Finset.univ a) → Eq (f a) 1
    -/
    /-
      🎉 no goals
    -/
    exact fun hc => (hc (Finset.mem_univ _)).elim
    /-
      🎉 no goals
    -/


/-- If a product of a `Finset` of a subsingleton type has a given
value, so do the terms in that product. -/
@[to_additive "If a sum of a `Finset` of a subsingleton type has a given
  value, so do the terms in that sum."]
theorem eq_of_subsingleton_of_prod_eq {ι : Type*} [Subsingleton ι] {s : Finset ι} {f : ι → M}
    {b : M} (h : ∏ i ∈ s, f i = b) : ∀ i ∈ s, f i = b :=
  Finset.eq_of_card_le_one_of_prod_eq (Finset.card_le_one_of_subsingleton s) h


@[to_additive (attr := simp)]
theorem Fintype.prod_option (f : Option α → M) : ∏ i, f i = f none * ∏ i, f (some i) :=
  Finset.prod_insertNone f univ


@[simp] lemma Finset.card_pi (s : Finset ι) (t : ∀ i, Finset (α i)) :
    #(s.pi t) = ∏ i ∈ s, #(t i) := Multiset.card_pi _ _


@[simp] lemma card_piFinset (s : ∀ i, Finset (α i)) :
                                      /-
                                        ι : Type u_4
                                        α : ι → Type u_6
                                        inst✝¹ : DecidableEq ι
                                        inst✝ : Fintype ι
                                        s : (i : ι) → Finset (α i)
                                        ⊢ Eq (Fintype.piFinset s).card (Finset.univ.prod fun i => (s i).card)
                                      -/
    #(piFinset s) = ∏ i, #(s i) := by simp [piFinset, card_map]
                                      /-
                                        🎉 no goals
                                      -/


/-- This lemma is specifically designed to be used backwards, whence the specialisation to `Fin n`
as the indexing type doesn't matter in practice. The more general forward direction lemma here is
`Fintype.card_piFinset`. -/
lemma card_piFinset_const {α : Type*} (s : Finset α) (n : ℕ) :
                                                 /-
                                                   α : Type u_7
                                                   s : Finset α
                                                   n : Nat
                                                   ⊢ Eq (Fintype.piFinset fun x => s).card (HPow.hPow s.card n)
                                                 -/
    #(piFinset fun _ : Fin n ↦ s) = #s ^ n := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp] lemma card_pi [∀ i, Fintype (α i)] : card (∀ i, α i) = ∏ i, card (α i) :=
  card_piFinset _


/-- This lemma is specifically designed to be used backwards, whence the specialisation to `Fin n`
as the indexing type doesn't matter in practice. The more general forward direction lemma here is
`Fintype.card_pi`. -/
lemma card_pi_const (α : Type*) [Fintype α] (n : ℕ) : card (Fin n → α) = card α ^ n :=
  card_piFinset_const _ _


@[simp] nonrec lemma card_sigma {ι} {α : ι → Type*} [Fintype ι] [∀ i, Fintype (α i)] :
    card (Sigma α) = ∑ i, card (α i) := card_sigma _ _


/-- The number of dependent maps `f : Π j, s j` for which the `i` component is `a` is the product
over all `j ≠ i` of `#(s j)`.

Note that this is just a composition of easier lemmas, but there's some glue missing to make that
smooth enough not to need this lemma. -/
lemma card_filter_piFinset_eq_of_mem [∀ i, DecidableEq (α i)]
    (s : ∀ i, Finset (α i)) (i : ι) {a : α i} (ha : a ∈ s i) :
    #{f ∈ piFinset s | f i = a} = ∏ j ∈ univ.erase i, #(s j) := by
  calc
    _ = ∏ j, #(Function.update s i {a} j) := by
      rw [← piFinset_update_singleton_eq_filter_piFinset_eq _ _ ha, Fintype.card_piFinset]
    _ = ∏ j, Function.update (fun j ↦ #(s j)) i 1 j :=
      Fintype.prod_congr _ _ fun j ↦ by obtain rfl | hji := eq_or_ne j i <;> simp [*]
    _ = _ := by simp [prod_update_of_mem, erase_eq]


lemma card_filter_piFinset_const_eq_of_mem (s : Finset κ) (i : ι) {x : κ} (hx : x ∈ s) :
    #{f ∈ piFinset fun _ ↦ s | f i = x} = #s ^ (card ι - 1) :=
  (card_filter_piFinset_eq_of_mem _ _ hx).trans <| by
    /-
      ι : Type u_4
      κ : Type u_5
      inst✝² : DecidableEq ι
      inst✝¹ : DecidableEq κ
      inst✝ : Fintype ι
      s : Finset κ
      i : ι
      x : κ
      hx : Membership.mem s x
      ⊢ Eq ((Finset.univ.erase i).prod fun j => s.card) (HPow.hPow s.card (HSub.hSub …
    -/
    rw [prod_const #s, card_erase_of_mem (mem_univ _), card_univ]
    /-
      🎉 no goals
    -/


lemma card_filter_piFinset_eq [∀ i, DecidableEq (α i)] (s : ∀ i, Finset (α i)) (i : ι) (a : α i) :
    #{f ∈ piFinset s | f i = a} = if a ∈ s i then ∏ b ∈ univ.erase i, #(s b) else 0 := by
  /-
    ι : Type u_4
    α : ι → Type u_6
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (α i)
    s : (i : ι) → Finset (α i)
    i : ι
    a : α i
    ⊢ Eq (Finset.filter (fun f => Eq (f i) a) (Fintype.piFinset s)).card (ite (Mem …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_4
      α : ι → Type u_6
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (α i)
      s : (i : ι) → Finset (α i)
      i : ι
      a : α i
      h : Membership.mem (s i) a
      ⊢ Eq (Finset.filter (fun f => Eq (f i) a) (Fintype.piFinset s)).card ((Finset. …
    -/
  · rw [card_filter_piFinset_eq_of_mem _ _ h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_4
      α : ι → Type u_6
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (α i)
      s : (i : ι) → Finset (α i)
      i : ι
      a : α i
      h : Not (Membership.mem (s i) a)
      ⊢ Eq (Finset.filter (fun f => Eq (f i) a) (Fintype.piFinset s)).card 0
    -/
  · rw [filter_piFinset_of_not_mem _ _ _ h, Finset.card_empty]
    /-
      🎉 no goals
    -/


lemma card_filter_piFinset_const (s : Finset κ) (i : ι) (j : κ) :
    #{f ∈ piFinset fun _ ↦ s | f i = j} = if j ∈ s then #s ^ (card ι - 1) else 0 :=
  (card_filter_piFinset_eq _ _ _).trans <| by
    /-
      ι : Type u_4
      κ : Type u_5
      inst✝² : DecidableEq ι
      inst✝¹ : DecidableEq κ
      inst✝ : Fintype ι
      s : Finset κ
      i : ι
      j : κ
      ⊢ Eq (ite (Membership.mem s j) ((Finset.univ.erase i).prod fun b => s.card) 0) …
    -/
    rw [prod_const #s, card_erase_of_mem (mem_univ _), card_univ]
    /-
      🎉 no goals
    -/


theorem Fintype.card_fun [DecidableEq α] [Fintype α] [Fintype β] :
    Fintype.card (α → β) = Fintype.card β ^ Fintype.card α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    ⊢ Eq (Fintype.card (α → β)) (HPow.hPow (Fintype.card β) (Fintype.card α))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem card_vector [Fintype α] (n : ℕ) :
    Fintype.card (List.Vector α n) = Fintype.card α ^ n := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    n : Nat
    ⊢ Eq (Fintype.card (List.Vector α n)) (HPow.hPow (Fintype.card α) n)
  -/
  rw [Fintype.ofEquiv_card]; simp
                             /-
                               🎉 no goals
                             -/


/-- It is equivalent to compute the product of a function over `Fin n` or `Finset.range n`. -/
@[to_additive "It is equivalent to sum a function over `fin n` or `finset.range n`."]
theorem Fin.prod_univ_eq_prod_range [CommMonoid α] (f : ℕ → α) (n : ℕ) :
    ∏ i : Fin n, f i = ∏ i ∈ range n, f i :=
  calc
    ∏ i : Fin n, f i = ∏ i : { x // x ∈ range n }, f i :=
                                                                              /-
                                                                                α : Type u_1
                                                                                inst✝ : CommMonoid α
                                                                                f : Nat → α
                                                                                n : Nat
                                                                                ⊢ ∀ (x : Nat), Iff (LT.lt x n) (Membership.mem (Finset.range n) x)
                                                                              -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
      Fintype.prod_equiv (Fin.equivSubtype.trans (Equiv.subtypeEquivRight (by simp))) _ _ (by simp)
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
                                 /-
                                   α : Type u_1
                                   inst✝ : CommMonoid α
                                   f : Nat → α
                                   n : Nat
                                   ⊢ Eq (Finset.univ.prod fun i => f ↑i) ((Finset.range n).prod fun i => f i)
                                 -/
    _ = ∏ i ∈ range n, f i := by rw [← attach_eq_univ, prod_attach]
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive]
theorem Finset.prod_fin_eq_prod_range [CommMonoid β] {n : ℕ} (c : Fin n → β) :
    ∏ i, c i = ∏ i ∈ Finset.range n, if h : i < n then c ⟨i, h⟩ else 1 := by
  /-
    β : Type u_2
    inst✝ : CommMonoid β
    n : Nat
    c : Fin n → β
    ⊢ Eq (Finset.univ.prod fun i => c i) ((Finset.range n).prod fun i => dite (LT. …
  -/
  rw [← Fin.prod_univ_eq_prod_range, Finset.prod_congr rfl]
  /-
    β : Type u_2
    inst✝ : CommMonoid β
    n : Nat
    c : Fin n → β
    ⊢ ∀ (x : Fin n), Membership.mem Finset.univ x → Eq (c x) (dite (LT.lt (↑x) n)  …
  -/
  rintro ⟨i, hi⟩ _
  /-
    case mk
    β : Type u_2
    inst✝ : CommMonoid β
    n : Nat
    c : Fin n → β
    i : Nat
    hi : LT.lt i n
    a✝ : Membership.mem Finset.univ ⟨i, hi⟩
    ⊢ Eq (c ⟨i, hi⟩) (dite (LT.lt (↑⟨i, hi⟩) n) (fun h => c ⟨↑⟨i, hi⟩, h⟩) fun h = …
  -/
  simp only [hi, dif_pos]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Finset.prod_toFinset_eq_subtype {M : Type*} [CommMonoid M] [Fintype α] (p : α → Prop)
    [DecidablePred p] (f : α → M) : ∏ a ∈ { x | p x }.toFinset, f a = ∏ a : Subtype p, f a := by
  /-
    α : Type u_1
    M : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    f : α → M
    ⊢ Eq ((setOf fun x => p x).toFinset.prod fun a => f a) (Finset.univ.prod fun a …
  -/
  rw [← Finset.prod_subtype]
  /-
    case h
    α : Type u_1
    M : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    f : α → M
    ⊢ ∀ (x : α), Iff (Membership.mem (setOf fun x => p x).toFinset x) (p x)
  -/
  simp_rw [Set.mem_toFinset]; intro; rfl
                                     /-
                                       🎉 no goals
                                     -/


nonrec theorem Fintype.prod_dite [Fintype α] {p : α → Prop} [DecidablePred p] [CommMonoid β]
    (f : ∀ a, p a → β) (g : ∀ a, ¬p a → β) :
    (∏ a, dite (p a) (f a) (g a)) =
    (∏ a : { a // p a }, f a a.2) * ∏ a : { a // ¬p a }, g a a.2 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : CommMonoid β
    f : (a : α) → p a → β
    g : (a : α) → Not (p a) → β
    ⊢ Eq (Finset.univ.prod fun a => dite (p a) (f a) (g a)) (HMul.hMul (Finset.uni …
  -/
  simp only [prod_dite, attach_eq_univ]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : CommMonoid β
    f : (a : α) → p a → β
    g : (a : α) → Not (p a) → β
    ⊢ Eq (HMul.hMul (Finset.univ.prod fun x => f ↑x ⋯) (Finset.univ.prod fun x =>  …
  -/
  congr 1
    /-
      case e_a
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : CommMonoid β
      f : (a : α) → p a → β
      g : (a : α) → Not (p a) → β
      ⊢ Eq (Finset.univ.prod fun x => f ↑x ⋯) (Finset.univ.prod fun a => f ↑a ⋯)
    -/
  · exact (Equiv.subtypeEquivRight <| by simp).prod_comp fun x : { x // p x } => f x x.2
    /-
      🎉 no goals
    -/
    /-
      case e_a
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : CommMonoid β
      f : (a : α) → p a → β
      g : (a : α) → Not (p a) → β
      ⊢ Eq (Finset.univ.prod fun x => g ↑x ⋯) (Finset.univ.prod fun a => g ↑a ⋯)
    -/
  · exact (Equiv.subtypeEquivRight <| by simp).prod_comp fun x : { x // ¬p x } => g x x.2
    /-
      🎉 no goals
    -/


@[to_additive]
theorem Fintype.prod_sum_elim (f : α₁ → M) (g : α₂ → M) :
    ∏ x, Sum.elim f g x = (∏ a₁, f a₁) * ∏ a₂, g a₂ :=
  prod_disj_sum _ _ _


@[to_additive (attr := simp)]
theorem Fintype.prod_sum_type (f : α₁ ⊕ α₂ → M) :
    ∏ x, f x = (∏ a₁, f (Sum.inl a₁)) * ∏ a₂, f (Sum.inr a₂) :=
  prod_disj_sum _ _ _


/-- The product over a product type equals the product of the fiberwise products. For rewriting
in the reverse direction, use `Fintype.prod_prod_type'`. -/
@[to_additive Fintype.sum_prod_type "The sum over a product type equals the sum of fiberwise sums.
For rewriting in the reverse direction, use `Fintype.sum_prod_type'`."]
theorem Fintype.prod_prod_type [CommMonoid γ] (f : α₁ × α₂ → γ) :
    ∏ x, f x = ∏ x, ∏ y, f (x, y) :=
  Finset.prod_product ..


/-- The product over a product type equals the product of the fiberwise products. For rewriting
in the reverse direction, use `Fintype.prod_prod_type`. -/
@[to_additive Fintype.sum_prod_type' "The sum over a product type equals the sum of fiberwise sums.
For rewriting in the reverse direction, use `Fintype.sum_prod_type`."]
theorem Fintype.prod_prod_type' [CommMonoid γ] (f : α₁ → α₂ → γ) :
    ∏ x : α₁ × α₂, f x.1 x.2 = ∏ x, ∏ y, f x y :=
  Finset.prod_product' ..


@[to_additive Fintype.sum_prod_type_right]
theorem Fintype.prod_prod_type_right [CommMonoid γ] (f : α₁ × α₂ → γ) :
    ∏ x, f x = ∏ y, ∏ x, f (x, y) :=
  Finset.prod_product_right ..


/-- An uncurried version of `Finset.prod_prod_type_right`. -/
@[to_additive Fintype.sum_prod_type_right' "An uncurried version of `Finset.sum_prod_type_right`"]
theorem Fintype.prod_prod_type_right' [CommMonoid γ] (f : α₁ → α₂ → γ) :
    ∏ x : α₁ × α₂, f x.1 x.2 = ∏ y, ∏ x, f x y :=
  Finset.prod_product_right' ..


