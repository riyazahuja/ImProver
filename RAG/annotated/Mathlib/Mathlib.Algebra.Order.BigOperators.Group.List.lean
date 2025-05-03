@[to_additive sum_le_sum]
lemma Forall₂.prod_le_prod' [Preorder M] [MulRightMono M]
    [MulLeftMono M] {l₁ l₂ : List M} (h : Forall₂ (· ≤ ·) l₁ l₂) :
    l₁.prod ≤ l₂.prod := by
  /-
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : Preorder M
    inst✝¹ : MulRightMono M
    inst✝ : MulLeftMono M
    l₁ l₂ : List M
    h : List.Forall₂ (fun x1 x2 => LE.le x1 x2) l₁ l₂
    ⊢ LE.le l₁.prod l₂.prod
  -/
  induction' h with a b la lb hab ih ih'
    /-
      case nil
      M : Type u_3
      inst✝³ : Monoid M
      inst✝² : Preorder M
      inst✝¹ : MulRightMono M
      inst✝ : MulLeftMono M
      l₁ l₂ : List M
      ⊢ LE.le List.nil.prod List.nil.prod
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      M : Type u_3
      inst✝³ : Monoid M
      inst✝² : Preorder M
      inst✝¹ : MulRightMono M
      inst✝ : MulLeftMono M
      l₁ l₂ : List M
      a b : M
      la lb : List M
      hab : LE.le a b
      ih : List.Forall₂ (fun x1 x2 => LE.le x1 x2) la lb
      ih' : LE.le la.prod lb.prod
      ⊢ LE.le (List.cons a la).prod (List.cons b lb).prod
    -/
  · simpa only [prod_cons] using mul_le_mul' hab ih'
    /-
      🎉 no goals
    -/


/-- If `l₁` is a sublist of `l₂` and all elements of `l₂` are greater than or equal to one, then
`l₁.prod ≤ l₂.prod`. One can prove a stronger version assuming `∀ a ∈ l₂.diff l₁, 1 ≤ a` instead
of `∀ a ∈ l₂, 1 ≤ a` but this lemma is not yet in `mathlib`. -/
@[to_additive sum_le_sum "If `l₁` is a sublist of `l₂` and all elements of `l₂` are nonnegative,
  then `l₁.sum ≤ l₂.sum`.
  One can prove a stronger version assuming `∀ a ∈ l₂.diff l₁, 0 ≤ a` instead of `∀ a ∈ l₂, 0 ≤ a`
  but this lemma is not yet in `mathlib`."]
lemma Sublist.prod_le_prod' [Preorder M] [MulRightMono M]
    [MulLeftMono M] {l₁ l₂ : List M} (h : l₁ <+ l₂)
    (h₁ : ∀ a ∈ l₂, (1 : M) ≤ a) : l₁.prod ≤ l₂.prod := by
  induction h with
  | slnil => rfl
  | cons a _ ih' =>
    simp only [prod_cons, forall_mem_cons] at h₁ ⊢
    exact (ih' h₁.2).trans (le_mul_of_one_le_left' h₁.1)
  | cons₂ a _ ih' =>
    simp only [prod_cons, forall_mem_cons] at h₁ ⊢
    exact mul_le_mul_left' (ih' h₁.2) _


@[to_additive sum_le_sum]
lemma SublistForall₂.prod_le_prod' [Preorder M]
    [MulRightMono M] [MulLeftMono M]
    {l₁ l₂ : List M} (h : SublistForall₂ (· ≤ ·) l₁ l₂) (h₁ : ∀ a ∈ l₂, (1 : M) ≤ a) :
    l₁.prod ≤ l₂.prod :=
  let ⟨_, hall, hsub⟩ := sublistForall₂_iff.1 h
  hall.prod_le_prod'.trans <| hsub.prod_le_prod' h₁


@[to_additive sum_le_sum]
lemma prod_le_prod' [Preorder M] [MulRightMono M]
    [MulLeftMono M] {l : List ι} {f g : ι → M} (h : ∀ i ∈ l, f i ≤ g i) :
    (l.map f).prod ≤ (l.map g).prod :=
                              /-
                                ι : Type u_1
                                M : Type u_3
                                inst✝³ : Monoid M
                                inst✝² : Preorder M
                                inst✝¹ : MulRightMono M
                                inst✝ : MulLeftMono M
                                l : List ι
                                f g : ι → M
                                h : ∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)
                                ⊢ List.Forall₂ (fun x1 x2 => LE.le x1 x2) (List.map f l) (List.map g l)
                              -/
  Forall₂.prod_le_prod' <| by simpa
                              /-
                                🎉 no goals
                              -/


@[to_additive sum_lt_sum]
lemma prod_lt_prod' [Preorder M] [MulLeftStrictMono M]
    [MulLeftMono M] [MulRightStrictMono M]
    [MulRightMono M] {l : List ι} (f g : ι → M)
    (h₁ : ∀ i ∈ l, f i ≤ g i) (h₂ : ∃ i ∈ l, f i < g i) : (l.map f).prod < (l.map g).prod := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝⁵ : Monoid M
    inst✝⁴ : Preorder M
    inst✝³ : MulLeftStrictMono M
    inst✝² : MulLeftMono M
    inst✝¹ : MulRightStrictMono M
    inst✝ : MulRightMono M
    l : List ι
    f g : ι → M
    h₁ : ∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)
    h₂ : Exists fun i => And (Membership.mem l i) (LT.lt (f i) (g i))
    ⊢ LT.lt (List.map f l).prod (List.map g l).prod
  -/
  induction' l with i l ihl
    /-
      case nil
      ι : Type u_1
      M : Type u_3
      inst✝⁵ : Monoid M
      inst✝⁴ : Preorder M
      inst✝³ : MulLeftStrictMono M
      inst✝² : MulLeftMono M
      inst✝¹ : MulRightStrictMono M
      inst✝ : MulRightMono M
      f g : ι → M
      h₁ : ∀ (i : ι), Membership.mem List.nil i → LE.le (f i) (g i)
      h₂ : Exists fun i => And (Membership.mem List.nil i) (LT.lt (f i) (g i))
      ⊢ LT.lt (List.map f List.nil).prod (List.map g List.nil).prod
    -/
  · rcases h₂ with ⟨_, ⟨⟩, _⟩
    /-
      🎉 no goals
    -/
  /-
    case cons
    ι : Type u_1
    M : Type u_3
    inst✝⁵ : Monoid M
    inst✝⁴ : Preorder M
    inst✝³ : MulLeftStrictMono M
    inst✝² : MulLeftMono M
    inst✝¹ : MulRightStrictMono M
    inst✝ : MulRightMono M
    f g : ι → M
    i : ι
    l : List ι
    ihl : (∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)) → (Exists fun i => A …
    h₁ : ∀ (i_1 : ι), Membership.mem (List.cons i l) i_1 → LE.le (f i_1) (g i_1)
    h₂ : Exists fun i_1 => And (Membership.mem (List.cons i l) i_1) (LT.lt (f i_1) …
    ⊢ LT.lt (List.map f (List.cons i l)).prod (List.map g (List.cons i l)).prod
  -/
  simp only [forall_mem_cons, map_cons, prod_cons] at h₁ ⊢
  /-
    case cons
    ι : Type u_1
    M : Type u_3
    inst✝⁵ : Monoid M
    inst✝⁴ : Preorder M
    inst✝³ : MulLeftStrictMono M
    inst✝² : MulLeftMono M
    inst✝¹ : MulRightStrictMono M
    inst✝ : MulRightMono M
    f g : ι → M
    i : ι
    l : List ι
    ihl : (∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)) → (Exists fun i => A …
    h₂ : Exists fun i_1 => And (Membership.mem (List.cons i l) i_1) (LT.lt (f i_1) …
    h₁ : And (LE.le (f i) (g i)) (∀ (x : ι), Membership.mem l x → LE.le (f x) (g x))
    ⊢ LT.lt (HMul.hMul (f i) (List.map f l).prod) (HMul.hMul (g i) (List.map g l). …
  -/
  simp only [mem_cons, exists_eq_or_imp] at h₂
  /-
    case cons
    ι : Type u_1
    M : Type u_3
    inst✝⁵ : Monoid M
    inst✝⁴ : Preorder M
    inst✝³ : MulLeftStrictMono M
    inst✝² : MulLeftMono M
    inst✝¹ : MulRightStrictMono M
    inst✝ : MulRightMono M
    f g : ι → M
    i : ι
    l : List ι
    ihl : (∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)) → (Exists fun i => A …
    h₁ : And (LE.le (f i) (g i)) (∀ (x : ι), Membership.mem l x → LE.le (f x) (g x))
    h₂ : Or (LT.lt (f i) (g i)) (Exists fun a => And (Membership.mem l a) (LT.lt ( …
    ⊢ LT.lt (HMul.hMul (f i) (List.map f l).prod) (HMul.hMul (g i) (List.map g l). …
  -/
  cases h₂
    /-
      case cons.inl
      ι : Type u_1
      M : Type u_3
      inst✝⁵ : Monoid M
      inst✝⁴ : Preorder M
      inst✝³ : MulLeftStrictMono M
      inst✝² : MulLeftMono M
      inst✝¹ : MulRightStrictMono M
      inst✝ : MulRightMono M
      f g : ι → M
      i : ι
      l : List ι
      ihl : (∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)) → (Exists fun i => A …
      h₁ : And (LE.le (f i) (g i)) (∀ (x : ι), Membership.mem l x → LE.le (f x) (g x))
      h✝ : LT.lt (f i) (g i)
      ⊢ LT.lt (HMul.hMul (f i) (List.map f l).prod) (HMul.hMul (g i) (List.map g l). …
    -/
  · exact mul_lt_mul_of_lt_of_le ‹_› (prod_le_prod' h₁.2)
    /-
      🎉 no goals
    -/
    /-
      case cons.inr
      ι : Type u_1
      M : Type u_3
      inst✝⁵ : Monoid M
      inst✝⁴ : Preorder M
      inst✝³ : MulLeftStrictMono M
      inst✝² : MulLeftMono M
      inst✝¹ : MulRightStrictMono M
      inst✝ : MulRightMono M
      f g : ι → M
      i : ι
      l : List ι
      ihl : (∀ (i : ι), Membership.mem l i → LE.le (f i) (g i)) → (Exists fun i => A …
      h₁ : And (LE.le (f i) (g i)) (∀ (x : ι), Membership.mem l x → LE.le (f x) (g x))
      h✝ : Exists fun a => And (Membership.mem l a) (LT.lt (f a) (g a))
      ⊢ LT.lt (HMul.hMul (f i) (List.map f l).prod) (HMul.hMul (g i) (List.map g l). …
    -/
  · exact mul_lt_mul_of_le_of_lt h₁.1 <| ihl h₁.2 ‹_›
    /-
      🎉 no goals
    -/


@[to_additive]
lemma prod_lt_prod_of_ne_nil [Preorder M] [MulLeftStrictMono M]
    [MulLeftMono M] [MulRightStrictMono M]
    [MulRightMono M] {l : List ι} (hl : l ≠ []) (f g : ι → M)
    (hlt : ∀ i ∈ l, f i < g i) : (l.map f).prod < (l.map g).prod :=
  (prod_lt_prod' f g fun i hi => (hlt i hi).le) <|
    (exists_mem_of_ne_nil l hl).imp fun i hi => ⟨hi, hlt i hi⟩


@[to_additive sum_le_card_nsmul]
lemma prod_le_pow_card [Preorder M] [MulRightMono M]
    [MulLeftMono M] (l : List M) (n : M) (h : ∀ x ∈ l, x ≤ n) :
    l.prod ≤ n ^ l.length := by
      /-
        M : Type u_3
        inst✝³ : Monoid M
        inst✝² : Preorder M
        inst✝¹ : MulRightMono M
        inst✝ : MulLeftMono M
        l : List M
        n : M
        h : ∀ (x : M), Membership.mem l x → LE.le x n
        ⊢ LE.le l.prod (HPow.hPow n l.length)
      -/
      simpa only [map_id', map_const', prod_replicate] using prod_le_prod' h
      /-
        🎉 no goals
      -/


@[to_additive card_nsmul_le_sum]
lemma pow_card_le_prod [Preorder M] [MulRightMono M]
    [MulLeftMono M] (l : List M) (n : M) (h : ∀ x ∈ l, n ≤ x) :
    n ^ l.length ≤ l.prod :=
  @prod_le_pow_card Mᵒᵈ _ _ _ _ l n h


@[to_additive exists_lt_of_sum_lt]
lemma exists_lt_of_prod_lt' [LinearOrder M] [MulRightMono M]
    [MulLeftMono M] {l : List ι} (f g : ι → M)
    (h : (l.map f).prod < (l.map g).prod) : ∃ i ∈ l, f i < g i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulRightMono M
    inst✝ : MulLeftMono M
    l : List ι
    f g : ι → M
    h : LT.lt (List.map f l).prod (List.map g l).prod
    ⊢ Exists fun i => And (Membership.mem l i) (LT.lt (f i) (g i))
  -/
  contrapose! h
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulRightMono M
    inst✝ : MulLeftMono M
    l : List ι
    f g : ι → M
    h : ∀ (i : ι), Membership.mem l i → LE.le (g i) (f i)
    ⊢ LE.le (List.map g l).prod (List.map f l).prod
  -/
  exact prod_le_prod' h
  /-
    🎉 no goals
  -/


@[to_additive exists_le_of_sum_le]
lemma exists_le_of_prod_le' [LinearOrder M] [MulLeftStrictMono M]
    [MulLeftMono M] [MulRightStrictMono M]
    [MulRightMono M] {l : List ι} (hl : l ≠ []) (f g : ι → M)
    (h : (l.map f).prod ≤ (l.map g).prod) : ∃ x ∈ l, f x ≤ g x := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝⁵ : Monoid M
    inst✝⁴ : LinearOrder M
    inst✝³ : MulLeftStrictMono M
    inst✝² : MulLeftMono M
    inst✝¹ : MulRightStrictMono M
    inst✝ : MulRightMono M
    l : List ι
    hl : Ne l List.nil
    f g : ι → M
    h : LE.le (List.map f l).prod (List.map g l).prod
    ⊢ Exists fun x => And (Membership.mem l x) (LE.le (f x) (g x))
  -/
  contrapose! h
  /-
    ι : Type u_1
    M : Type u_3
    inst✝⁵ : Monoid M
    inst✝⁴ : LinearOrder M
    inst✝³ : MulLeftStrictMono M
    inst✝² : MulLeftMono M
    inst✝¹ : MulRightStrictMono M
    inst✝ : MulRightMono M
    l : List ι
    hl : Ne l List.nil
    f g : ι → M
    h : ∀ (x : ι), Membership.mem l x → LT.lt (g x) (f x)
    ⊢ LT.lt (List.map g l).prod (List.map f l).prod
  -/
  exact prod_lt_prod_of_ne_nil hl _ _ h
  /-
    🎉 no goals
  -/


@[to_additive sum_nonneg]
lemma one_le_prod_of_one_le [Preorder M] [MulLeftMono M] {l : List M}
    (hl₁ : ∀ x ∈ l, (1 : M) ≤ x) : 1 ≤ l.prod := by
  -- We don't use `pow_card_le_prod` to avoid assumption
  -- [covariant_class M M (function.swap (*)) (≤)]
  /-
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulLeftMono M
    l : List M
    hl₁ : ∀ (x : M), Membership.mem l x → LE.le 1 x
    ⊢ LE.le 1 l.prod
  -/
  induction' l with hd tl ih
    /-
      case nil
      M : Type u_3
      inst✝² : Monoid M
      inst✝¹ : Preorder M
      inst✝ : MulLeftMono M
      hl₁ : ∀ (x : M), Membership.mem List.nil x → LE.le 1 x
      ⊢ LE.le 1 List.nil.prod
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case cons
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulLeftMono M
    hd : M
    tl : List M
    ih : (∀ (x : M), Membership.mem tl x → LE.le 1 x) → LE.le 1 tl.prod
    hl₁ : ∀ (x : M), Membership.mem (List.cons hd tl) x → LE.le 1 x
    ⊢ LE.le 1 (List.cons hd tl).prod
  -/
  rw [prod_cons]
  /-
    case cons
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulLeftMono M
    hd : M
    tl : List M
    ih : (∀ (x : M), Membership.mem tl x → LE.le 1 x) → LE.le 1 tl.prod
    hl₁ : ∀ (x : M), Membership.mem (List.cons hd tl) x → LE.le 1 x
    ⊢ LE.le 1 (HMul.hMul hd tl.prod)
  -/
  exact one_le_mul (hl₁ hd (mem_cons_self hd tl)) (ih fun x h => hl₁ x (mem_cons_of_mem hd h))
  /-
    🎉 no goals
  -/


@[to_additive]
lemma max_prod_le (l : List α) (f g : α → M) [LinearOrder M]
    [MulLeftMono M] [MulRightMono M] :
    max (l.map f).prod (l.map g).prod ≤ (l.map fun i ↦ max (f i) (g i)).prod := by
  /-
    α : Type u_2
    M : Type u_3
    inst✝³ : Monoid M
    l : List α
    f g : α → M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    ⊢ LE.le (Max.max (List.map f l).prod (List.map g l).prod) (List.map (fun i =>  …
  -/
  rw [max_le_iff]
  /-
    α : Type u_2
    M : Type u_3
    inst✝³ : Monoid M
    l : List α
    f g : α → M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    ⊢ And (LE.le (List.map f l).prod (List.map (fun i => Max.max (f i) (g i)) l).p …
  -/
  constructor <;> apply List.prod_le_prod' <;> intros
    /-
      case left.h
      α : Type u_2
      M : Type u_3
      inst✝³ : Monoid M
      l : List α
      f g : α → M
      inst✝² : LinearOrder M
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      i✝ : α
      a✝ : Membership.mem l i✝
      ⊢ LE.le (f i✝) (Max.max (f i✝) (g i✝))
    -/
  · apply le_max_left
    /-
      🎉 no goals
    -/
    /-
      case right.h
      α : Type u_2
      M : Type u_3
      inst✝³ : Monoid M
      l : List α
      f g : α → M
      inst✝² : LinearOrder M
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      i✝ : α
      a✝ : Membership.mem l i✝
      ⊢ LE.le (g i✝) (Max.max (f i✝) (g i✝))
    -/
  · apply le_max_right
    /-
      🎉 no goals
    -/


@[to_additive]
lemma prod_min_le [LinearOrder M] [MulLeftMono M]
    [MulRightMono M] (l : List α) (f g : α → M) :
    (l.map fun i ↦ min (f i) (g i)).prod ≤ min (l.map f).prod (l.map g).prod := by
  /-
    α : Type u_2
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    l : List α
    f g : α → M
    ⊢ LE.le (List.map (fun i => Min.min (f i) (g i)) l).prod (Min.min (List.map f  …
  -/
  rw [le_min_iff]
  /-
    α : Type u_2
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    l : List α
    f g : α → M
    ⊢ And (LE.le (List.map (fun i => Min.min (f i) (g i)) l).prod (List.map f l).p …
  -/
  constructor <;> apply List.prod_le_prod' <;> intros
    /-
      case left.h
      α : Type u_2
      M : Type u_3
      inst✝³ : Monoid M
      inst✝² : LinearOrder M
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      l : List α
      f g : α → M
      i✝ : α
      a✝ : Membership.mem l i✝
      ⊢ LE.le (Min.min (f i✝) (g i✝)) (f i✝)
    -/
  · apply min_le_left
    /-
      🎉 no goals
    -/
    /-
      case right.h
      α : Type u_2
      M : Type u_3
      inst✝³ : Monoid M
      inst✝² : LinearOrder M
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      l : List α
      f g : α → M
      i✝ : α
      a✝ : Membership.mem l i✝
      ⊢ LE.le (Min.min (f i✝) (g i✝)) (g i✝)
    -/
  · apply min_le_right
    /-
      🎉 no goals
    -/


lemma sum_le_foldr_max [AddMonoid M] [AddMonoid N] [LinearOrder N] (f : M → N) (h0 : f 0 ≤ 0)
    (hadd : ∀ x y, f (x + y) ≤ max (f x) (f y)) (l : List M) : f l.sum ≤ (l.map f).foldr max 0 := by
  /-
    M : Type u_3
    N : Type u_4
    inst✝² : AddMonoid M
    inst✝¹ : AddMonoid N
    inst✝ : LinearOrder N
    f : M → N
    h0 : LE.le (f 0) 0
    hadd : ∀ (x y : M), LE.le (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
    l : List M
    ⊢ LE.le (f l.sum) (List.foldr Max.max 0 (List.map f l))
  -/
  induction' l with hd tl IH
    /-
      case nil
      M : Type u_3
      N : Type u_4
      inst✝² : AddMonoid M
      inst✝¹ : AddMonoid N
      inst✝ : LinearOrder N
      f : M → N
      h0 : LE.le (f 0) 0
      hadd : ∀ (x y : M), LE.le (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
      ⊢ LE.le (f List.nil.sum) (List.foldr Max.max 0 (List.map f List.nil))
    -/
  · simpa using h0
    /-
      🎉 no goals
    -/
  /-
    case cons
    M : Type u_3
    N : Type u_4
    inst✝² : AddMonoid M
    inst✝¹ : AddMonoid N
    inst✝ : LinearOrder N
    f : M → N
    h0 : LE.le (f 0) 0
    hadd : ∀ (x y : M), LE.le (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
    hd : M
    tl : List M
    IH : LE.le (f tl.sum) (List.foldr Max.max 0 (List.map f tl))
    ⊢ LE.le (f (List.cons hd tl).sum) (List.foldr Max.max 0 (List.map f (List.cons …
  -/
  simp only [List.sum_cons, List.foldr_map, List.foldr] at IH ⊢
  /-
    case cons
    M : Type u_3
    N : Type u_4
    inst✝² : AddMonoid M
    inst✝¹ : AddMonoid N
    inst✝ : LinearOrder N
    f : M → N
    h0 : LE.le (f 0) 0
    hadd : ∀ (x y : M), LE.le (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
    hd : M
    tl : List M
    IH : LE.le (f tl.sum) (List.foldr (fun x y => Max.max (f x) y) 0 tl)
    ⊢ LE.le (f (HAdd.hAdd hd tl.sum)) (Max.max (f hd) (List.foldr (fun x y => Max. …
  -/
  exact (hadd _ _).trans (max_le_max le_rfl IH)
  /-
    🎉 no goals
  -/


@[to_additive sum_pos]
lemma one_lt_prod_of_one_lt [OrderedCommMonoid M] :
    ∀ l : List M, (∀ x ∈ l, (1 : M) < x) → l ≠ [] → 1 < l.prod
  | [], _, h => (h rfl).elim
                    /-
                      M : Type u_3
                      inst✝ : OrderedCommMonoid M
                      b : M
                      h : ∀ (x : M), Membership.mem (List.cons b List.nil) x → LT.lt 1 x
                      x✝ : Ne (List.cons b List.nil) List.nil
                      ⊢ LT.lt 1 (List.cons b List.nil).prod
                    -/
  | [b], h, _ => by simpa using h
                    /-
                      🎉 no goals
                    -/
  | a :: b :: l, hl₁, _ => by
    /-
      M : Type u_3
      inst✝ : OrderedCommMonoid M
      a b : M
      l : List M
      hl₁ : ∀ (x : M), Membership.mem (List.cons a (List.cons b l)) x → LT.lt 1 x
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ LT.lt 1 (List.cons a (List.cons b l)).prod
    -/
    simp only [forall_eq_or_imp, List.mem_cons] at hl₁
    /-
      M : Type u_3
      inst✝ : OrderedCommMonoid M
      a b : M
      l : List M
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      hl₁ : And (LT.lt 1 a) (And (LT.lt 1 b) (∀ (a : M), Membership.mem l a → LT.lt  …
      ⊢ LT.lt 1 (List.cons a (List.cons b l)).prod
    -/
    rw [List.prod_cons]
    /-
      M : Type u_3
      inst✝ : OrderedCommMonoid M
      a b : M
      l : List M
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      hl₁ : And (LT.lt 1 a) (And (LT.lt 1 b) (∀ (a : M), Membership.mem l a → LT.lt  …
      ⊢ LT.lt 1 (HMul.hMul a (List.cons b l).prod)
    -/
    apply one_lt_mul_of_lt_of_le' hl₁.1
    /-
      M : Type u_3
      inst✝ : OrderedCommMonoid M
      a b : M
      l : List M
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      hl₁ : And (LT.lt 1 a) (And (LT.lt 1 b) (∀ (a : M), Membership.mem l a → LT.lt  …
      ⊢ LE.le 1 (List.cons b l).prod
    -/
    apply le_of_lt ((b :: l).one_lt_prod_of_one_lt _ (l.cons_ne_nil b))
    /-
      M : Type u_3
      inst✝ : OrderedCommMonoid M
      a b : M
      l : List M
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      hl₁ : And (LT.lt 1 a) (And (LT.lt 1 b) (∀ (a : M), Membership.mem l a → LT.lt  …
      ⊢ ∀ (x : M), Membership.mem (List.cons b l) x → LT.lt 1 x
    -/
    intro x hx; cases hx
      /-
        case head
        M : Type u_3
        inst✝ : OrderedCommMonoid M
        a b : M
        l : List M
        x✝ : Ne (List.cons a (List.cons b l)) List.nil
        hl₁ : And (LT.lt 1 a) (And (LT.lt 1 b) (∀ (a : M), Membership.mem l a → LT.lt  …
        ⊢ LT.lt 1 b
      -/
    · exact hl₁.2.1
      /-
        🎉 no goals
      -/
      /-
        case tail
        M : Type u_3
        inst✝ : OrderedCommMonoid M
        a b : M
        l : List M
        x✝ : Ne (List.cons a (List.cons b l)) List.nil
        hl₁ : And (LT.lt 1 a) (And (LT.lt 1 b) (∀ (a : M), Membership.mem l a → LT.lt  …
        x : M
        a✝ : List.Mem x l
        ⊢ LT.lt 1 x
      -/
    · exact hl₁.2.2 _ ‹_›
      /-
        🎉 no goals
      -/


/-- See also `List.le_prod_of_mem`. -/
@[to_additive "See also `List.le_sum_of_mem`."]
lemma single_le_prod [OrderedCommMonoid M] {l : List M} (hl₁ : ∀ x ∈ l, (1 : M) ≤ x) :
    ∀ x ∈ l, x ≤ l.prod := by
  /-
    M : Type u_3
    inst✝ : OrderedCommMonoid M
    l : List M
    hl₁ : ∀ (x : M), Membership.mem l x → LE.le 1 x
    ⊢ ∀ (x : M), Membership.mem l x → LE.le x l.prod
  -/
  induction l
    /-
      case nil
      M : Type u_3
      inst✝ : OrderedCommMonoid M
      hl₁ : ∀ (x : M), Membership.mem List.nil x → LE.le 1 x
      ⊢ ∀ (x : M), Membership.mem List.nil x → LE.le x List.nil.prod
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    M : Type u_3
    inst✝ : OrderedCommMonoid M
    head✝ : M
    tail✝ : List M
    tail_ih✝ : (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x) → ∀ (x : M), Member …
    hl₁ : ∀ (x : M), Membership.mem (List.cons head✝ tail✝) x → LE.le 1 x
    ⊢ ∀ (x : M), Membership.mem (List.cons head✝ tail✝) x → LE.le x (List.cons hea …
  -/
  simp_rw [prod_cons, forall_mem_cons] at hl₁ ⊢
  /-
    case cons
    M : Type u_3
    inst✝ : OrderedCommMonoid M
    head✝ : M
    tail✝ : List M
    tail_ih✝ : (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x) → ∀ (x : M), Member …
    hl₁ : And (LE.le 1 head✝) (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x)
    ⊢ And (LE.le head✝ (HMul.hMul head✝ tail✝.prod)) (∀ (x : M), Membership.mem ta …
  -/
  constructor
  /-
    case cons.left
    M : Type u_3
    inst✝ : OrderedCommMonoid M
    head✝ : M
    tail✝ : List M
    tail_ih✝ : (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x) → ∀ (x : M), Member …
    hl₁ : And (LE.le 1 head✝) (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x)
    ⊢ LE.le head✝ (HMul.hMul head✝ tail✝.prod)
  -/
  case cons.left => exact le_mul_of_one_le_right' (one_le_prod_of_one_le hl₁.2)
  /-
    case cons.right
    M : Type u_3
    inst✝ : OrderedCommMonoid M
    head✝ : M
    tail✝ : List M
    tail_ih✝ : (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x) → ∀ (x : M), Member …
    hl₁ : And (LE.le 1 head✝) (∀ (x : M), Membership.mem tail✝ x → LE.le 1 x)
    ⊢ ∀ (x : M), Membership.mem tail✝ x → LE.le x (HMul.hMul head✝ tail✝.prod)
  -/
  case cons.right hd tl ih => exact fun x H => le_mul_of_one_le_of_le hl₁.1 (ih hl₁.right x H)
  /-
    🎉 no goals
  -/


@[to_additive all_zero_of_le_zero_le_of_sum_eq_zero]
lemma all_one_of_le_one_le_of_prod_eq_one [OrderedCommMonoid M] {l : List M}
    (hl₁ : ∀ x ∈ l, (1 : M) ≤ x) (hl₂ : l.prod = 1) {x : M} (hx : x ∈ l) : x = 1 :=
  _root_.le_antisymm (hl₂ ▸ single_le_prod hl₁ _ hx) (hl₁ x hx)


@[to_additive] lemma prod_eq_one_iff : l.prod = 1 ↔ ∀ x ∈ l, x = (1 : M) :=
  ⟨all_one_of_le_one_le_of_prod_eq_one fun _ _ => one_le _, fun h => by
    /-
      M : Type u_3
      inst✝ : CanonicallyOrderedCommMonoid M
      l : List M
      h : ∀ (x : M), Membership.mem l x → Eq x 1
      ⊢ Eq l.prod 1
    -/
    rw [List.eq_replicate_iff.2 ⟨_, h⟩, prod_replicate, one_pow]
      /-
        M : Type u_3
        inst✝ : CanonicallyOrderedCommMonoid M
        l : List M
        h : ∀ (x : M), Membership.mem l x → Eq x 1
        ⊢ Nat
      -/
    · exact (length l)
      /-
        🎉 no goals
      -/
      /-
        M : Type u_3
        inst✝ : CanonicallyOrderedCommMonoid M
        l : List M
        h : ∀ (x : M), Membership.mem l x → Eq x 1
        ⊢ Eq l.length l.length
      -/
    · rfl⟩
      /-
        🎉 no goals
      -/


@[to_additive] lemma monotone_prod_take (L : List M) : Monotone fun i => (L.take i).prod := by
  /-
    M : Type u_3
    inst✝ : CanonicallyOrderedCommMonoid M
    L : List M
    ⊢ Monotone fun i => (List.take i L).prod
  -/
  refine monotone_nat_of_le_succ fun n => ?_
  /-
    M : Type u_3
    inst✝ : CanonicallyOrderedCommMonoid M
    L : List M
    n : Nat
    ⊢ LE.le (List.take n L).prod (List.take (HAdd.hAdd n 1) L).prod
  -/
  cases' lt_or_le n L.length with h h
    /-
      case inl
      M : Type u_3
      inst✝ : CanonicallyOrderedCommMonoid M
      L : List M
      n : Nat
      h : LT.lt n L.length
      ⊢ LE.le (List.take n L).prod (List.take (HAdd.hAdd n 1) L).prod
    -/
  · rw [prod_take_succ _ _ h]
    /-
      case inl
      M : Type u_3
      inst✝ : CanonicallyOrderedCommMonoid M
      L : List M
      n : Nat
      h : LT.lt n L.length
      ⊢ LE.le (List.take n L).prod (HMul.hMul (List.take n L).prod (GetElem.getElem  …
    -/
    exact le_self_mul
    /-
      🎉 no goals
    -/
    /-
      case inr
      M : Type u_3
      inst✝ : CanonicallyOrderedCommMonoid M
      L : List M
      n : Nat
      h : LE.le L.length n
      ⊢ LE.le (List.take n L).prod (List.take (HAdd.hAdd n 1) L).prod
    -/
  · simp [take_of_length_le h, take_of_length_le (le_trans h (Nat.le_succ _))]
    /-
      🎉 no goals
    -/


/-- See also `List.single_le_prod`. -/
@[to_additive "See also `List.single_le_sum`."]
theorem le_prod_of_mem {xs : List M} {x : M} (h₁ : x ∈ xs) : x ≤ xs.prod := by
  induction xs with
  | nil => simp at h₁
  | cons y ys ih =>
    simp only [mem_cons] at h₁
    rcases h₁ with (rfl | h₁)
    · simp
    · specialize ih h₁
      simp only [List.prod_cons]
      exact le_mul_left ih


