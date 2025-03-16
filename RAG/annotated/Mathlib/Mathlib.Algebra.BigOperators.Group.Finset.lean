/-- `∏ x ∈ s, f x` is the product of `f x` as `x` ranges over the elements of the finite set `s`.

When the index type is a `Fintype`, the notation `∏ x, f x`, is a shorthand for
`∏ x ∈ Finset.univ, f x`. -/
@[to_additive "`∑ x ∈ s, f x` is the sum of `f x` as `x` ranges over the elements
of the finite set `s`.

When the index type is a `Fintype`, the notation `∑ x, f x`, is a shorthand for
`∑ x ∈ Finset.univ, f x`."]
protected def prod [CommMonoid β] (s : Finset α) (f : α → β) : β :=
  (s.1.map f).prod


@[to_additive (attr := simp)]
theorem prod_mk [CommMonoid β] (s : Multiset α) (hs : s.Nodup) (f : α → β) :
    (⟨s, hs⟩ : Finset α).prod f = (s.map f).prod :=
  rfl


@[to_additive (attr := simp)]
theorem prod_val [CommMonoid α] (s : Finset α) : s.1.prod = s.prod id := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s : Finset α
    ⊢ Eq s.val.prod (s.prod id)
  -/
  rw [Finset.prod, Multiset.map_id]
  /-
    🎉 no goals
  -/


/-- A `bigOpBinder` is like an `extBinder` and has the form `x`, `x : ty`, or `x pred`
where `pred` is a `binderPred` like `< 2`.
Unlike `extBinder`, `x` is a term. -/
syntax bigOpBinder := term:max ((" : " term) <|> binderPred)?

/-- A BigOperator binder in parentheses -/
syntax bigOpBinderParenthesized := " (" bigOpBinder ")"

/-- A list of parenthesized binders -/
syntax bigOpBinderCollection := bigOpBinderParenthesized+

/-- A single (unparenthesized) binder, or a list of parenthesized binders -/
syntax bigOpBinders := bigOpBinderCollection <|> (ppSpace bigOpBinder)


/-- Collects additional binder/Finset pairs for the given `bigOpBinder`.
Note: this is not extensible at the moment, unlike the usual `bigOpBinder` expansions. -/
def processBigOpBinder (processed : (Array (Term × Term)))
    (binder : TSyntax ``bigOpBinder) : MacroM (Array (Term × Term)) :=
  set_option hygiene false in
  withRef binder do
    match binder with
    | `(bigOpBinder| $x:term) =>
      match x with
      | `(($a + $b = $n)) => -- Maybe this is too cute.
        return processed |>.push (← `(⟨$a, $b⟩), ← `(Finset.Nat.antidiagonal $n))
      | _ => return processed |>.push (x, ← ``(Finset.univ))
    | `(bigOpBinder| $x : $t) => return processed |>.push (x, ← ``((Finset.univ : Finset $t)))
    | `(bigOpBinder| $x ∈ $s) => return processed |>.push (x, ← `(finset% $s))
    | `(bigOpBinder| $x < $n) => return processed |>.push (x, ← `(Finset.Iio $n))
    | `(bigOpBinder| $x ≤ $n) => return processed |>.push (x, ← `(Finset.Iic $n))
    | `(bigOpBinder| $x > $n) => return processed |>.push (x, ← `(Finset.Ioi $n))
    | `(bigOpBinder| $x ≥ $n) => return processed |>.push (x, ← `(Finset.Ici $n))
    | _ => Macro.throwUnsupported


/-- Collects the binder/Finset pairs for the given `bigOpBinders`. -/
def processBigOpBinders (binders : TSyntax ``bigOpBinders) :
    MacroM (Array (Term × Term)) :=
  match binders with
  | `(bigOpBinders| $b:bigOpBinder) => processBigOpBinder #[] b
  | `(bigOpBinders| $[($bs:bigOpBinder)]*) => bs.foldlM processBigOpBinder #[]
  | _ => Macro.throwUnsupported


/-- Collect the binderIdents into a `⟨...⟩` expression. -/
def bigOpBindersPattern (processed : (Array (Term × Term))) :
    MacroM Term := do
  let ts := processed.map Prod.fst
  if ts.size == 1 then
    return ts[0]!
  else
    `(⟨$ts,*⟩)


/-- Collect the terms into a product of sets. -/
def bigOpBindersProd (processed : (Array (Term × Term))) :
    MacroM Term := do
  if processed.isEmpty then
    `((Finset.univ : Finset Unit))
  else if processed.size == 1 then
    return processed[0]!.2
  else
    processed.foldrM (fun s p => `(SProd.sprod $(s.2) $p)) processed.back!.2
      (start := processed.size - 1)


/--
- `∑ x, f x` is notation for `Finset.sum Finset.univ f`. It is the sum of `f x`,
  where `x` ranges over the finite domain of `f`.
- `∑ x ∈ s, f x` is notation for `Finset.sum s f`. It is the sum of `f x`,
  where `x` ranges over the finite set `s` (either a `Finset` or a `Set` with a `Fintype` instance).
- `∑ x ∈ s with p x, f x` is notation for `Finset.sum (Finset.filter p s) f`.
- `∑ (x ∈ s) (y ∈ t), f x y` is notation for `Finset.sum (s ×ˢ t) (fun ⟨x, y⟩ ↦ f x y)`.

These support destructuring, for example `∑ ⟨x, y⟩ ∈ s ×ˢ t, f x y`.

Notation: `"∑" bigOpBinders* ("with" term)? "," term` -/
syntax (name := bigsum) "∑ " bigOpBinders ("with " term)? ", " term:67 : term


/--
- `∏ x, f x` is notation for `Finset.prod Finset.univ f`. It is the product of `f x`,
  where `x` ranges over the finite domain of `f`.
- `∏ x ∈ s, f x` is notation for `Finset.prod s f`. It is the product of `f x`,
  where `x` ranges over the finite set `s` (either a `Finset` or a `Set` with a `Fintype` instance).
- `∏ x ∈ s with p x, f x` is notation for `Finset.prod (Finset.filter p s) f`.
- `∏ (x ∈ s) (y ∈ t), f x y` is notation for `Finset.prod (s ×ˢ t) (fun ⟨x, y⟩ ↦ f x y)`.

These support destructuring, for example `∏ ⟨x, y⟩ ∈ s ×ˢ t, f x y`.

Notation: `"∏" bigOpBinders* ("with" term)? "," term` -/
syntax (name := bigprod) "∏ " bigOpBinders ("with " term)? ", " term:67 : term


macro_rules (kind := bigsum)
  | `(∑ $bs:bigOpBinders $[with $p?]?, $v) => do
    let processed ← processBigOpBinders bs
    let x ← bigOpBindersPattern processed
    let s ← bigOpBindersProd processed
    match p? with
    | some p => `(Finset.sum (Finset.filter (fun $x ↦ $p) $s) (fun $x ↦ $v))
    | none => `(Finset.sum $s (fun $x ↦ $v))


macro_rules (kind := bigprod)
  | `(∏ $bs:bigOpBinders $[with $p?]?, $v) => do
    let processed ← processBigOpBinders bs
    let x ← bigOpBindersPattern processed
    let s ← bigOpBindersProd processed
    match p? with
    | some p => `(Finset.prod (Finset.filter (fun $x ↦ $p) $s) (fun $x ↦ $v))
    | none => `(Finset.prod $s (fun $x ↦ $v))


/-- (Deprecated, use `∑ x ∈ s, f x`)
`∑ x in s, f x` is notation for `Finset.sum s f`. It is the sum of `f x`,
where `x` ranges over the finite set `s`. -/
syntax (name := bigsumin) "∑ " extBinder " in " term ", " term:67 : term

macro_rules (kind := bigsumin)
  | `(∑ $x:ident in $s, $r) => `(∑ $x:ident ∈ $s, $r)
  | `(∑ $x:ident : $t in $s, $r) => `(∑ $x:ident ∈ ($s : Finset $t), $r)


/-- (Deprecated, use `∏ x ∈ s, f x`)
`∏ x in s, f x` is notation for `Finset.prod s f`. It is the product of `f x`,
where `x` ranges over the finite set `s`. -/
syntax (name := bigprodin) "∏ " extBinder " in " term ", " term:67 : term

macro_rules (kind := bigprodin)
  | `(∏ $x:ident in $s, $r) => `(∏ $x:ident ∈ $s, $r)
  | `(∏ $x:ident : $t in $s, $r) => `(∏ $x:ident ∈ ($s : Finset $t), $r)


/-- Delaborator for `Finset.prod`. The `pp.piBinderTypes` option controls whether
to show the domain type when the product is over `Finset.univ`. -/
@[app_delab Finset.prod] def delabFinsetProd : Delab :=
  whenPPOption getPPNotation <| withOverApp 5 <| do
  let #[_, _, _, s, f] := (← getExpr).getAppArgs | failure
  guard <| f.isLambda
  let ppDomain ← getPPOption getPPPiBinderTypes
  let (i, body) ← withAppArg <| withBindingBodyUnusedName fun i => do
    return (i, ← delab)
  if s.isAppOfArity ``Finset.univ 2 then
    let binder ←
      if ppDomain then
        let ty ← withNaryArg 0 delab
        `(bigOpBinder| $(.mk i):ident : $ty)
      else
        `(bigOpBinder| $(.mk i):ident)
    `(∏ $binder:bigOpBinder, $body)
  else
    let ss ← withNaryArg 3 <| delab
    `(∏ $(.mk i):ident ∈ $ss, $body)


/-- Delaborator for `Finset.sum`. The `pp.piBinderTypes` option controls whether
to show the domain type when the sum is over `Finset.univ`. -/
@[app_delab Finset.sum] def delabFinsetSum : Delab :=
  whenPPOption getPPNotation <| withOverApp 5 <| do
  let #[_, _, _, s, f] := (← getExpr).getAppArgs | failure
  guard <| f.isLambda
  let ppDomain ← getPPOption getPPPiBinderTypes
  let (i, body) ← withAppArg <| withBindingBodyUnusedName fun i => do
    return (i, ← delab)
  if s.isAppOfArity ``Finset.univ 2 then
    let binder ←
      if ppDomain then
        let ty ← withNaryArg 0 delab
        `(bigOpBinder| $(.mk i):ident : $ty)
      else
        `(bigOpBinder| $(.mk i):ident)
    `(∑ $binder:bigOpBinder, $body)
  else
    let ss ← withNaryArg 3 <| delab
    `(∑ $(.mk i):ident ∈ $ss, $body)


@[to_additive]
theorem prod_eq_multiset_prod [CommMonoid β] (s : Finset α) (f : α → β) :
    ∏ x ∈ s, f x = (s.1.map f).prod :=
  rfl


@[to_additive (attr := simp)]
lemma prod_map_val [CommMonoid β] (s : Finset α) (f : α → β) : (s.1.map f).prod = ∏ a ∈ s, f a :=
  rfl


@[to_additive]
theorem prod_eq_fold [CommMonoid β] (s : Finset α) (f : α → β) :
    ∏ x ∈ s, f x = s.fold ((· * ·) : β → β → β) 1 f :=
  rfl


@[simp]
theorem sum_multiset_singleton (s : Finset α) : (s.sum fun x => {x}) = s.val := by
  /-
    α : Type u_3
    s : Finset α
    ⊢ Eq (s.sum fun x => Singleton.singleton x) s.val
  -/
  simp only [sum_eq_multiset_sum, Multiset.sum_map_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem map_prod [CommMonoid β] [CommMonoid γ] {G : Type*} [FunLike G β γ] [MonoidHomClass G β γ]
    (g : G) (f : α → β) (s : Finset α) : g (∏ x ∈ s, f x) = ∏ x ∈ s, g (f x) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝³ : CommMonoid β
    inst✝² : CommMonoid γ
    G : Type u_6
    inst✝¹ : FunLike G β γ
    inst✝ : MonoidHomClass G β γ
    g : G
    f : α → β
    s : Finset α
    ⊢ Eq (g (s.prod fun x => f x)) (s.prod fun x => g (f x))
  -/
  simp only [Finset.prod_eq_multiset_prod, map_multiset_prod, Multiset.map_map]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[to_additive]
theorem MonoidHom.coe_finset_prod [MulOneClass β] [CommMonoid γ] (f : α → β →* γ) (s : Finset α) :
    ⇑(∏ x ∈ s, f x) = ∏ x ∈ s, ⇑(f x) :=
  map_prod (MonoidHom.coeFn β γ) _ _


/-- See also `Finset.prod_apply`, with the same conclusion but with the weaker hypothesis
`f : α → β → γ` -/
@[to_additive (attr := simp)
  "See also `Finset.sum_apply`, with the same conclusion but with the weaker hypothesis
  `f : α → β → γ`"]
theorem MonoidHom.finset_prod_apply [MulOneClass β] [CommMonoid γ] (f : α → β →* γ) (s : Finset α)
    (b : β) : (∏ x ∈ s, f x) b = ∏ x ∈ s, f x b :=
  map_prod (MonoidHom.eval b) _ _


@[to_additive (attr := simp)]
theorem prod_empty : ∏ x ∈ ∅, f x = 1 :=
  rfl


@[to_additive]
theorem prod_of_isEmpty [IsEmpty α] (s : Finset α) : ∏ i ∈ s, f i = 1 := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝¹ : CommMonoid β
    inst✝ : IsEmpty α
    s : Finset α
    ⊢ Eq (s.prod fun i => f i) 1
  -/
  rw [eq_empty_of_isEmpty s, prod_empty]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-11")] alias prod_of_empty := prod_of_isEmpty

@[deprecated (since := "2024-06-11")] alias sum_of_empty := sum_of_isEmpty


@[to_additive (attr := simp)]
theorem prod_cons (h : a ∉ s) : ∏ x ∈ cons a s h, f x = f a * ∏ x ∈ s, f x :=
  fold_cons h


@[to_additive (attr := simp)]
theorem prod_insert [DecidableEq α] : a ∉ s → ∏ x ∈ insert a s, f x = f a * ∏ x ∈ s, f x :=
  fold_insert


/-- The product of `f` over `insert a s` is the same as
the product over `s`, as long as `a` is in `s` or `f a = 1`. -/
@[to_additive (attr := simp) "The sum of `f` over `insert a s` is the same as
the sum over `s`, as long as `a` is in `s` or `f a = 0`."]
theorem prod_insert_of_eq_one_if_not_mem [DecidableEq α] (h : a ∉ s → f a = 1) :
    ∏ x ∈ insert a s, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    a : α
    f : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    h : Not (Membership.mem s a) → Eq (f a) 1
    ⊢ Eq ((Insert.insert a s).prod fun x => f x) (s.prod fun x => f x)
  -/
  by_cases hm : a ∈ s
    /-
      case pos
      α : Type u_3
      β : Type u_4
      s : Finset α
      a : α
      f : α → β
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      h : Not (Membership.mem s a) → Eq (f a) 1
      hm : Membership.mem s a
      ⊢ Eq ((Insert.insert a s).prod fun x => f x) (s.prod fun x => f x)
    -/
  · simp_rw [insert_eq_of_mem hm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      s : Finset α
      a : α
      f : α → β
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      h : Not (Membership.mem s a) → Eq (f a) 1
      hm : Not (Membership.mem s a)
      ⊢ Eq ((Insert.insert a s).prod fun x => f x) (s.prod fun x => f x)
    -/
  · rw [prod_insert hm, h hm, one_mul]
    /-
      🎉 no goals
    -/


/-- The product of `f` over `insert a s` is the same as
the product over `s`, as long as `f a = 1`. -/
@[to_additive (attr := simp) "The sum of `f` over `insert a s` is the same as
the sum over `s`, as long as `f a = 0`."]
theorem prod_insert_one [DecidableEq α] (h : f a = 1) : ∏ x ∈ insert a s, f x = ∏ x ∈ s, f x :=
  prod_insert_of_eq_one_if_not_mem fun _ => h


@[to_additive]
theorem prod_insert_div {M : Type*} [CommGroup M] [DecidableEq α] (ha : a ∉ s) {f : α → M} :
                                                       /-
                                                         α : Type u_3
                                                         s : Finset α
                                                         a : α
                                                         M : Type u_6
                                                         inst✝¹ : CommGroup M
                                                         inst✝ : DecidableEq α
                                                         ha : Not (Membership.mem s a)
                                                         f : α → M
                                                         ⊢ Eq (HDiv.hDiv ((Insert.insert a s).prod fun x => f x) (f a)) (s.prod fun x = …
                                                       -/
    (∏ x ∈ insert a s, f x) / f a = ∏ x ∈ s, f x := by simp [ha]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive (attr := simp)]
theorem prod_singleton (f : α → β) (a : α) : ∏ x ∈ singleton a, f x = f a :=
  Eq.trans fold_singleton <| mul_one _


@[to_additive]
theorem prod_pair [DecidableEq α] {a b : α} (h : a ≠ b) :
    (∏ x ∈ ({a, b} : Finset α), f x) = f a * f b := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    a b : α
    h : Ne a b
    ⊢ Eq ((Insert.insert a (Singleton.singleton b)).prod fun x => f x) (HMul.hMul  …
  -/
  rw [prod_insert (not_mem_singleton.2 h), prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_const_one : (∏ _x ∈ s, (1 : β)) = 1 := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝ : CommMonoid β
    ⊢ Eq (s.prod fun _x => 1) 1
  -/
  simp only [Finset.prod, Multiset.map_const', Multiset.prod_replicate, one_pow]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_image [DecidableEq α] {s : Finset γ} {g : γ → α} :
    (∀ x ∈ s, ∀ y ∈ s, g x = g y → x = y) → ∏ x ∈ s.image g, f x = ∏ x ∈ s, f (g x) :=
  fold_image


@[to_additive (attr := simp)]
theorem prod_map (s : Finset α) (e : α ↪ γ) (f : γ → β) :
    ∏ x ∈ s.map e, f x = ∏ x ∈ s, f (e x) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    s : Finset α
    e : Function.Embedding α γ
    f : γ → β
    ⊢ Eq ((Finset.map e s).prod fun x => f x) (s.prod fun x => f (e x))
  -/
  rw [Finset.prod, Finset.map_val, Multiset.map_map]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
lemma prod_attach (s : Finset α) (f : α → β) : ∏ x ∈ s.attach, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    ⊢ Eq (s.attach.prod fun x => f ↑x) (s.prod fun x => f x)
  -/
  classical rw [← prod_image Subtype.coe_injective.injOn, attach_image_val]
  /-
    🎉 no goals
  -/


@[to_additive (attr := congr)]
theorem prod_congr (h : s₁ = s₂) : (∀ x ∈ s₂, f x = g x) → s₁.prod f = s₂.prod g := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f g : α → β
    inst✝ : CommMonoid β
    h : Eq s₁ s₂
    ⊢ (∀ (x : α), Membership.mem s₂ x → Eq (f x) (g x)) → Eq (s₁.prod f) (s₂.prod g)
  -/
  rw [h]; exact fold_congr
          /-
            🎉 no goals
          -/


@[to_additive]
theorem prod_eq_one {f : α → β} {s : Finset α} (h : ∀ x ∈ s, f x = 1) : ∏ x ∈ s, f x = 1 :=
  calc
    ∏ x ∈ s, f x = ∏ _x ∈ s, 1 := Finset.prod_congr rfl h
    _ = 1 := Finset.prod_const_one


@[to_additive]
theorem prod_disjUnion (h) :
    ∏ x ∈ s₁.disjUnion s₂ h, f x = (∏ x ∈ s₁, f x) * ∏ x ∈ s₂, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝ : CommMonoid β
    h : Disjoint s₁ s₂
    ⊢ Eq ((s₁.disjUnion s₂ h).prod fun x => f x) (HMul.hMul (s₁.prod fun x => f x) …
  -/
  refine Eq.trans ?_ (fold_disjUnion h)
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝ : CommMonoid β
    h : Disjoint s₁ s₂
    ⊢ Eq ((s₁.disjUnion s₂ h).prod fun x => f x) (Finset.fold (fun x1 x2 => HMul.h …
  -/
  rw [one_mul]
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝ : CommMonoid β
    h : Disjoint s₁ s₂
    ⊢ Eq ((s₁.disjUnion s₂ h).prod fun x => f x) (Finset.fold (fun x1 x2 => HMul.h …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_disjiUnion (s : Finset ι) (t : ι → Finset α) (h) :
    ∏ x ∈ s.disjiUnion t h, f x = ∏ i ∈ s, ∏ x ∈ t i, f x := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝ : CommMonoid β
    s : Finset ι
    t : ι → Finset α
    h : (↑s).PairwiseDisjoint t
    ⊢ Eq ((s.disjiUnion t h).prod fun x => f x) (s.prod fun i => (t i).prod fun x  …
  -/
  refine Eq.trans ?_ (fold_disjiUnion h)
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝ : CommMonoid β
    s : Finset ι
    t : ι → Finset α
    h : (↑s).PairwiseDisjoint t
    ⊢ Eq ((s.disjiUnion t h).prod fun x => f x) (Finset.fold (fun x1 x2 => HMul.hM …
  -/
  dsimp [Finset.prod, Multiset.prod, Multiset.fold, Finset.disjUnion, Finset.fold]
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝ : CommMonoid β
    s : Finset ι
    t : ι → Finset α
    h : (↑s).PairwiseDisjoint t
    ⊢ Eq (Multiset.foldr (fun x1 x2 => HMul.hMul x1 x2) 1 (Multiset.map (fun x =>  …
  -/
  congr
  /-
    case e_b
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝ : CommMonoid β
    s : Finset ι
    t : ι → Finset α
    h : (↑s).PairwiseDisjoint t
    ⊢ Eq 1 (Multiset.foldr (fun x1 x2 => HMul.hMul x1 x2) 1 (Multiset.map (fun i = …
  -/
  exact prod_const_one.symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_union_inter [DecidableEq α] :
    (∏ x ∈ s₁ ∪ s₂, f x) * ∏ x ∈ s₁ ∩ s₂, f x = (∏ x ∈ s₁, f x) * ∏ x ∈ s₂, f x :=
  fold_union_inter


@[to_additive]
theorem prod_union [DecidableEq α] (h : Disjoint s₁ s₂) :
    ∏ x ∈ s₁ ∪ s₂, f x = (∏ x ∈ s₁, f x) * ∏ x ∈ s₂, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    h : Disjoint s₁ s₂
    ⊢ Eq ((Union.union s₁ s₂).prod fun x => f x) (HMul.hMul (s₁.prod fun x => f x) …
  -/
  rw [← prod_union_inter, disjoint_iff_inter_eq_empty.mp h]; exact (mul_one _).symm
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
theorem prod_filter_mul_prod_filter_not
    (s : Finset α) (p : α → Prop) [DecidablePred p] [∀ x, Decidable (¬p x)] (f : α → β) :
    (∏ x ∈ s with p x, f x) * ∏ x ∈ s with ¬p x, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝² : CommMonoid β
    s : Finset α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : (x : α) → Decidable (Not (p x))
    f : α → β
    ⊢ Eq (HMul.hMul ((Finset.filter (fun x => p x) s).prod fun x => f x) ((Finset. …
  -/
  have := Classical.decEq α
  /-
    α : Type u_3
    β : Type u_4
    inst✝² : CommMonoid β
    s : Finset α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : (x : α) → Decidable (Not (p x))
    f : α → β
    this : DecidableEq α
    ⊢ Eq (HMul.hMul ((Finset.filter (fun x => p x) s).prod fun x => f x) ((Finset. …
  -/
  rw [← prod_union (disjoint_filter_filter_neg s s p), filter_union_filter_neg_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_filter_not_mul_prod_filter (s : Finset α) (p : α → Prop) [DecidablePred p]
    [∀ x, Decidable (¬p x)] (f : α → β) :
    (∏ x ∈ s.filter fun x ↦ ¬p x, f x) * ∏ x ∈ s.filter p, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝² : CommMonoid β
    s : Finset α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : (x : α) → Decidable (Not (p x))
    f : α → β
    ⊢ Eq (HMul.hMul ((Finset.filter (fun x => Not (p x)) s).prod fun x => f x) ((F …
  -/
  rw [mul_comm, prod_filter_mul_prod_filter_not]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_filter_xor (p q : α → Prop) [DecidablePred p] [DecidablePred q] :
    (∏ x ∈ s with (Xor' (p x) (q x)), f x) =
      (∏ x ∈ s with (p x ∧ ¬ q x), f x) * (∏ x ∈ s with (q x ∧ ¬ p x), f x) := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝² : CommMonoid β
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    ⊢ Eq ((Finset.filter (fun x => Xor' (p x) (q x)) s).prod fun x => f x) (HMul.h …
  -/
  classical rw [← prod_union (disjoint_filter_and_not_filter _ _), ← filter_or]
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝² : CommMonoid β
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    ⊢ Eq ((Finset.filter (fun x => Xor' (p x) (q x)) s).prod fun x => f x) ((Finse …
  -/
  simp only [Xor']
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_to_list (s : Finset α) (f : α → β) : (s.toList.map f).prod = s.prod f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    ⊢ Eq (List.map f s.toList).prod (s.prod f)
  -/
  rw [Finset.prod, ← Multiset.prod_coe, ← Multiset.map_coe, Finset.coe_toList]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem _root_.Equiv.Perm.prod_comp (σ : Equiv.Perm α) (s : Finset α) (f : α → β)
    (hs : { a | σ a ≠ a } ⊆ s) : (∏ x ∈ s, f (σ x)) = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    σ : Equiv.Perm α
    s : Finset α
    f : α → β
    hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) ↑s
    ⊢ Eq (s.prod fun x => f (σ x)) (s.prod fun x => f x)
  -/
  convert (prod_map s σ.toEmbedding f).symm
  /-
    case h.e'_3.h
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    σ : Equiv.Perm α
    s : Finset α
    f : α → β
    hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) ↑s
    ⊢ Eq s (Finset.map (Equiv.toEmbedding σ) s)
  -/
  exact (map_perm hs).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem _root_.Equiv.Perm.prod_comp' (σ : Equiv.Perm α) (s : Finset α) (f : α → α → β)
    (hs : { a | σ a ≠ a } ⊆ s) : (∏ x ∈ s, f (σ x) x) = ∏ x ∈ s, f x (σ.symm x) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    σ : Equiv.Perm α
    s : Finset α
    f : α → α → β
    hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) ↑s
    ⊢ Eq (s.prod fun x => f (σ x) x) (s.prod fun x => f x ((Equiv.symm σ) x))
  -/
  convert σ.prod_comp s (fun x => f x (σ.symm x)) hs
  /-
    case h.e'_2.a.h.e'_2
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    σ : Equiv.Perm α
    s : Finset α
    f : α → α → β
    hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) ↑s
    x✝ : α
    a✝ : Membership.mem s x✝
    ⊢ Eq x✝ ((Equiv.symm σ) (σ x✝))
  -/
  rw [Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- A product over all subsets of `s ∪ {x}` is obtained by multiplying the product over all subsets
of `s`, and over all subsets of `s` to which one adds `x`. -/
@[to_additive "A sum over all subsets of `s ∪ {x}` is obtained by summing the sum over all subsets
of `s`, and over all subsets of `s` to which one adds `x`."]
lemma prod_powerset_insert [DecidableEq α] (ha : a ∉ s) (f : Finset α → β) :
    ∏ t ∈ (insert a s).powerset, f t =
      (∏ t ∈ s.powerset, f t) * ∏ t ∈ s.powerset, f (insert a t) := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    a : α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    f : Finset α → β
    ⊢ Eq ((Insert.insert a s).powerset.prod fun t => f t) (HMul.hMul (s.powerset.p …
  -/
  rw [powerset_insert, prod_union, prod_image]
    /-
      α : Type u_3
      β : Type u_4
      s : Finset α
      a : α
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      f : Finset α → β
      ⊢ ∀ (x : Finset α), Membership.mem s.powerset x → ∀ (y : Finset α), Membership …
    -/
  · exact insert_erase_invOn.2.injOn.mono fun t ht ↦ not_mem_mono (mem_powerset.1 ht) ha
    /-
      🎉 no goals
    -/
    /-
      α : Type u_3
      β : Type u_4
      s : Finset α
      a : α
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      f : Finset α → β
      ⊢ Disjoint s.powerset (Finset.image (Insert.insert a) s.powerset)
    -/
  · aesop (add simp [disjoint_left, insert_subset_iff])
    /-
      🎉 no goals
    -/


/-- A product over all subsets of `s ∪ {x}` is obtained by multiplying the product over all subsets
of `s`, and over all subsets of `s` to which one adds `x`. -/
@[to_additive "A sum over all subsets of `s ∪ {x}` is obtained by summing the sum over all subsets
of `s`, and over all subsets of `s` to which one adds `x`."]
lemma prod_powerset_cons (ha : a ∉ s) (f : Finset α → β) :
    ∏ t ∈ (s.cons a ha).powerset, f t = (∏ t ∈ s.powerset, f t) *
      ∏ t ∈ s.powerset.attach, f (cons a t <| not_mem_mono (mem_powerset.1 t.2) ha) := by
  classical
  simp_rw [cons_eq_insert]
  rw [prod_powerset_insert ha, prod_attach _ fun t ↦ f (insert a t)]


/-- A product over `powerset s` is equal to the double product over sets of subsets of `s` with
`#s = k`, for `k = 1, ..., #s`. -/
@[to_additive "A sum over `powerset s` is equal to the double sum over sets of subsets of `s` with
`#s = k`, for `k = 1, ..., #s`"]
lemma prod_powerset (s : Finset α) (f : Finset α → β) :
    ∏ t ∈ powerset s, f t = ∏ j ∈ range (#s + 1), ∏ t ∈ powersetCard j s, f t := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : Finset α → β
    ⊢ Eq (s.powerset.prod fun t => f t) ((Finset.range (HAdd.hAdd s.card 1)).prod  …
  -/
  rw [powerset_card_disjiUnion, prod_disjiUnion]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCompl.prod_mul_prod {s t : Finset α} (h : IsCompl s t) (f : α → β) :
    (∏ i ∈ s, f i) * ∏ i ∈ t, f i = ∏ i, f i :=
  (Finset.prod_disjUnion h.disjoint).symm.trans <| by
    /-
      α : Type u_3
      β : Type u_4
      inst✝¹ : Fintype α
      inst✝ : CommMonoid β
      s t : Finset α
      h : IsCompl s t
      f : α → β
      ⊢ Eq ((s.disjUnion t ⋯).prod fun x => f x) (Finset.univ.prod fun i => f i)
    -/
    classical rw [Finset.disjUnion_eq_union, ← Finset.sup_eq_union, h.sup_eq_top]; rfl
    /-
      🎉 no goals
    -/


/-- Multiplying the products of a function over `s` and over `sᶜ` gives the whole product.
For a version expressed with subtypes, see `Fintype.prod_subtype_mul_prod_subtype`. -/
@[to_additive "Adding the sums of a function over `s` and over `sᶜ` gives the whole sum.
For a version expressed with subtypes, see `Fintype.sum_subtype_add_sum_subtype`. "]
theorem prod_mul_prod_compl [Fintype α] [DecidableEq α] (s : Finset α) (f : α → β) :
    (∏ i ∈ s, f i) * ∏ i ∈ sᶜ, f i = ∏ i, f i :=
  IsCompl.prod_mul_prod isCompl_compl f


@[to_additive]
theorem prod_compl_mul_prod [Fintype α] [DecidableEq α] (s : Finset α) (f : α → β) :
    (∏ i ∈ sᶜ, f i) * ∏ i ∈ s, f i = ∏ i, f i :=
  (@isCompl_compl _ s _).symm.prod_mul_prod f


@[to_additive]
theorem prod_sdiff [DecidableEq α] (h : s₁ ⊆ s₂) :
    (∏ x ∈ s₂ \ s₁, f x) * ∏ x ∈ s₁, f x = ∏ x ∈ s₂, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    h : HasSubset.Subset s₁ s₂
    ⊢ Eq (HMul.hMul ((SDiff.sdiff s₂ s₁).prod fun x => f x) (s₁.prod fun x => f x) …
  -/
  rw [← prod_union sdiff_disjoint, sdiff_union_of_subset h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_subset_one_on_sdiff [DecidableEq α] (h : s₁ ⊆ s₂) (hg : ∀ x ∈ s₂ \ s₁, g x = 1)
    (hfg : ∀ x ∈ s₁, f x = g x) : ∏ i ∈ s₁, f i = ∏ i ∈ s₂, g i := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f g : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    h : HasSubset.Subset s₁ s₂
    hg : ∀ (x : α), Membership.mem (SDiff.sdiff s₂ s₁) x → Eq (g x) 1
    hfg : ∀ (x : α), Membership.mem s₁ x → Eq (f x) (g x)
    ⊢ Eq (s₁.prod fun i => f i) (s₂.prod fun i => g i)
  -/
  rw [← prod_sdiff h, prod_eq_one hg, one_mul]
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f g : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    h : HasSubset.Subset s₁ s₂
    hg : ∀ (x : α), Membership.mem (SDiff.sdiff s₂ s₁) x → Eq (g x) 1
    hfg : ∀ (x : α), Membership.mem s₁ x → Eq (f x) (g x)
    ⊢ Eq (s₁.prod fun i => f i) (s₁.prod fun x => g x)
  -/
  exact prod_congr rfl hfg
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_subset (h : s₁ ⊆ s₂) (hf : ∀ x ∈ s₂, x ∉ s₁ → f x = 1) :
    ∏ x ∈ s₁, f x = ∏ x ∈ s₂, f x :=
  haveI := Classical.decEq α
                                 /-
                                   α : Type u_3
                                   β : Type u_4
                                   s₁ s₂ : Finset α
                                   f : α → β
                                   inst✝ : CommMonoid β
                                   h : HasSubset.Subset s₁ s₂
                                   hf : ∀ (x : α), Membership.mem s₂ x → Not (Membership.mem s₁ x) → Eq (f x) 1
                                   this : DecidableEq α
                                   ⊢ ∀ (x : α), Membership.mem (SDiff.sdiff s₂ s₁) x → Eq (f x) 1
                                 -/
  prod_subset_one_on_sdiff h (by simpa) fun _ _ => rfl
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive (attr := simp)]
theorem prod_disj_sum (s : Finset α) (t : Finset γ) (f : α ⊕ γ → β) :
    ∏ x ∈ s.disjSum t, f x = (∏ x ∈ s, f (Sum.inl x)) * ∏ x ∈ t, f (Sum.inr x) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    s : Finset α
    t : Finset γ
    f : Sum α γ → β
    ⊢ Eq ((s.disjSum t).prod fun x => f x) (HMul.hMul (s.prod fun x => f (Sum.inl  …
  -/
  rw [← map_inl_disjUnion_map_inr, prod_disjUnion, prod_map, prod_map]
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    s : Finset α
    t : Finset γ
    f : Sum α γ → β
    ⊢ Eq (HMul.hMul (s.prod fun x => f (Function.Embedding.inl x)) (t.prod fun x = …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_sum_elim (s : Finset α) (t : Finset γ) (f : α → β) (g : γ → β) :
                                                                            /-
                                                                              α : Type u_3
                                                                              β : Type u_4
                                                                              γ : Type u_5
                                                                              inst✝ : CommMonoid β
                                                                              s : Finset α
                                                                              t : Finset γ
                                                                              f : α → β
                                                                              g : γ → β
                                                                              ⊢ Eq ((s.disjSum t).prod fun x => Sum.elim f g x) (HMul.hMul (s.prod fun x =>  …
                                                                            -/
    ∏ x ∈ s.disjSum t, Sum.elim f g x = (∏ x ∈ s, f x) * ∏ x ∈ t, g x := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[to_additive]
theorem prod_biUnion [DecidableEq α] {s : Finset γ} {t : γ → Finset α}
    (hs : Set.PairwiseDisjoint (↑s) t) : ∏ x ∈ s.biUnion t, f x = ∏ x ∈ s, ∏ i ∈ t x, f i := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    f : α → β
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset γ
    t : γ → Finset α
    hs : (↑s).PairwiseDisjoint t
    ⊢ Eq ((s.biUnion t).prod fun x => f x) (s.prod fun x => (t x).prod fun i => f i)
  -/
  rw [← disjiUnion_eq_biUnion _ _ hs, prod_disjiUnion]
  /-
    🎉 no goals
  -/


/-- The product over a sigma type equals the product of the fiberwise products. For rewriting
in the reverse direction, use `Finset.prod_sigma'`. -/
@[to_additive "The sum over a sigma type equals the sum of the fiberwise sums. For rewriting
in the reverse direction, use `Finset.sum_sigma'`"]
theorem prod_sigma {σ : α → Type*} (s : Finset α) (t : ∀ a, Finset (σ a)) (f : Sigma σ → β) :
    ∏ x ∈ s.sigma t, f x = ∏ a ∈ s, ∏ s ∈ t a, f ⟨a, s⟩ := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    σ : α → Type u_6
    s : Finset α
    t : (a : α) → Finset (σ a)
    f : Sigma σ → β
    ⊢ Eq ((s.sigma t).prod fun x => f x) (s.prod fun a => (t a).prod fun s => f ⟨a …
  -/
  simp_rw [← disjiUnion_map_sigma_mk, prod_disjiUnion, prod_map, Function.Embedding.sigmaMk_apply]
  /-
    🎉 no goals
  -/


/-- The product over a sigma type equals the product of the fiberwise products. For rewriting
in the reverse direction, use `Finset.prod_sigma`. -/
@[to_additive "The sum over a sigma type equals the sum of the fiberwise sums. For rewriting
in the reverse direction, use `Finset.sum_sigma`"]
theorem prod_sigma' {σ : α → Type*} (s : Finset α) (t : ∀ a, Finset (σ a)) (f : ∀ a, σ a → β) :
    (∏ a ∈ s, ∏ s ∈ t a, f a s) = ∏ x ∈ s.sigma t, f x.1 x.2 :=
  Eq.symm <| prod_sigma s t fun x => f x.1 x.2


/-- Reorder a product.

The difference with `Finset.prod_bij'` is that the bijection is specified as a surjective injection,
rather than by an inverse function.

The difference with `Finset.prod_nbij` is that the bijection is allowed to use membership of the
domain of the product, rather than being a non-dependent function. -/
@[to_additive "Reorder a sum.

The difference with `Finset.sum_bij'` is that the bijection is specified as a surjective injection,
rather than by an inverse function.

The difference with `Finset.sum_nbij` is that the bijection is allowed to use membership of the
domain of the sum, rather than being a non-dependent function."]
theorem prod_bij (i : ∀ a ∈ s, κ) (hi : ∀ a ha, i a ha ∈ t)
    (i_inj : ∀ a₁ ha₁ a₂ ha₂, i a₁ ha₁ = i a₂ ha₂ → a₁ = a₂)
    (i_surj : ∀ b ∈ t, ∃ a ha, i a ha = b) (h : ∀ a ha, f a = g (i a ha)) :
    ∏ x ∈ s, f x = ∏ x ∈ t, g x :=
  congr_arg Multiset.prod (Multiset.map_eq_map_of_bij_of_nodup f g s.2 t.2 i hi i_inj i_surj h)


/-- Reorder a product.

The difference with `Finset.prod_bij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.prod_nbij'` is that the bijection and its inverse are allowed to use
membership of the domains of the products, rather than being non-dependent functions. -/
@[to_additive "Reorder a sum.

The difference with `Finset.sum_bij` is that the bijection is specified with an inverse, rather than
as a surjective injection.

The difference with `Finset.sum_nbij'` is that the bijection and its inverse are allowed to use
membership of the domains of the sums, rather than being non-dependent functions."]
theorem prod_bij' (i : ∀ a ∈ s, κ) (j : ∀ a ∈ t, ι) (hi : ∀ a ha, i a ha ∈ t)
    (hj : ∀ a ha, j a ha ∈ s) (left_inv : ∀ a ha, j (i a ha) (hi a ha) = a)
    (right_inv : ∀ a ha, i (j a ha) (hj a ha) = a) (h : ∀ a ha, f a = g (i a ha)) :
    ∏ x ∈ s, f x = ∏ x ∈ t, g x := by
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝ : CommMonoid α
    s : Finset ι
    t : Finset κ
    f : ι → α
    g : κ → α
    i : (a : ι) → Membership.mem s a → κ
    j : (a : κ) → Membership.mem t a → ι
    hi : ∀ (a : ι) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : κ) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : ι) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : κ) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    h : ∀ (a : ι) (ha : Membership.mem s a), Eq (f a) (g (i a ha))
    ⊢ Eq (s.prod fun x => f x) (t.prod fun x => g x)
  -/
  refine prod_bij i hi (fun a1 h1 a2 h2 eq ↦ ?_) (fun b hb ↦ ⟨_, hj b hb, right_inv b hb⟩) h
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝ : CommMonoid α
    s : Finset ι
    t : Finset κ
    f : ι → α
    g : κ → α
    i : (a : ι) → Membership.mem s a → κ
    j : (a : κ) → Membership.mem t a → ι
    hi : ∀ (a : ι) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : κ) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : ι) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : κ) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    h : ∀ (a : ι) (ha : Membership.mem s a), Eq (f a) (g (i a ha))
    a1 : ι
    h1 : Membership.mem s a1
    a2 : ι
    h2 : Membership.mem s a2
    eq : Eq (i a1 h1) (i a2 h2)
    ⊢ Eq a1 a2
  -/
  rw [← left_inv a1 h1, ← left_inv a2 h2]
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝ : CommMonoid α
    s : Finset ι
    t : Finset κ
    f : ι → α
    g : κ → α
    i : (a : ι) → Membership.mem s a → κ
    j : (a : κ) → Membership.mem t a → ι
    hi : ∀ (a : ι) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : κ) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : ι) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : κ) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    h : ∀ (a : ι) (ha : Membership.mem s a), Eq (f a) (g (i a ha))
    a1 : ι
    h1 : Membership.mem s a1
    a2 : ι
    h2 : Membership.mem s a2
    eq : Eq (i a1 h1) (i a2 h2)
    ⊢ Eq (j (i a1 h1) ⋯) (j (i a2 h2) ⋯)
  -/
  simp only [eq]
  /-
    🎉 no goals
  -/


/-- Reorder a product.

The difference with `Finset.prod_nbij'` is that the bijection is specified as a surjective
injection, rather than by an inverse function.

The difference with `Finset.prod_bij` is that the bijection is a non-dependent function, rather than
being allowed to use membership of the domain of the product. -/
@[to_additive "Reorder a sum.

The difference with `Finset.sum_nbij'` is that the bijection is specified as a surjective injection,
rather than by an inverse function.

The difference with `Finset.sum_bij` is that the bijection is a non-dependent function, rather than
being allowed to use membership of the domain of the sum."]
lemma prod_nbij (i : ι → κ) (hi : ∀ a ∈ s, i a ∈ t) (i_inj : (s : Set ι).InjOn i)
    (i_surj : (s : Set ι).SurjOn i t) (h : ∀ a ∈ s, f a = g (i a)) :
    ∏ x ∈ s, f x = ∏ x ∈ t, g x :=
                                        /-
                                          ι : Type u_6
                                          κ : Type u_7
                                          α : Type u_8
                                          inst✝ : CommMonoid α
                                          s : Finset ι
                                          t : Finset κ
                                          f : ι → α
                                          g : κ → α
                                          i : ι → κ
                                          hi : ∀ (a : ι), Membership.mem s a → Membership.mem t (i a)
                                          i_inj : Set.InjOn i ↑s
                                          i_surj : Set.SurjOn i ↑s ↑t
                                          h : ∀ (a : ι), Membership.mem s a → Eq (f a) (g (i a))
                                          ⊢ ∀ (b : κ), Membership.mem t b → Exists fun a => Exists fun ha => Eq ((fun a  …
                                        -/
  prod_bij (fun a _ ↦ i a) hi i_inj (by simpa using i_surj) h
                                        /-
                                          🎉 no goals
                                        -/


/-- Reorder a product.

The difference with `Finset.prod_nbij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.prod_bij'` is that the bijection and its inverse are non-dependent
functions, rather than being allowed to use membership of the domains of the products.

The difference with `Finset.prod_equiv` is that bijectivity is only required to hold on the domains
of the products, rather than on the entire types.
-/
@[to_additive "Reorder a sum.

The difference with `Finset.sum_nbij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.sum_bij'` is that the bijection and its inverse are non-dependent
functions, rather than being allowed to use membership of the domains of the sums.

The difference with `Finset.sum_equiv` is that bijectivity is only required to hold on the domains
of the sums, rather than on the entire types."]
lemma prod_nbij' (i : ι → κ) (j : κ → ι) (hi : ∀ a ∈ s, i a ∈ t) (hj : ∀ a ∈ t, j a ∈ s)
    (left_inv : ∀ a ∈ s, j (i a) = a) (right_inv : ∀ a ∈ t, i (j a) = a)
    (h : ∀ a ∈ s, f a = g (i a)) : ∏ x ∈ s, f x = ∏ x ∈ t, g x :=
  prod_bij' (fun a _ ↦ i a) (fun b _ ↦ j b) hi hj left_inv right_inv h


/-- Specialization of `Finset.prod_nbij'` that automatically fills in most arguments.

See `Fintype.prod_equiv` for the version where `s` and `t` are `univ`. -/
@[to_additive "`Specialization of `Finset.sum_nbij'` that automatically fills in most arguments.

See `Fintype.sum_equiv` for the version where `s` and `t` are `univ`."]
lemma prod_equiv (e : ι ≃ κ) (hst : ∀ i, i ∈ s ↔ e i ∈ t) (hfg : ∀ i ∈ s, f i = g (e i)) :
                                      /-
                                        ι : Type u_6
                                        κ : Type u_7
                                        α : Type u_8
                                        inst✝ : CommMonoid α
                                        s : Finset ι
                                        t : Finset κ
                                        f : ι → α
                                        g : κ → α
                                        e : Equiv ι κ
                                        hst : ∀ (i : ι), Iff (Membership.mem s i) (Membership.mem t (e i))
                                        hfg : ∀ (i : ι), Membership.mem s i → Eq (f i) (g (e i))
                                        ⊢ Eq (s.prod fun i => f i) (t.prod fun i => g i)
                                      -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    ∏ i ∈ s, f i = ∏ i ∈ t, g i := by refine prod_nbij' e e.symm ?_ ?_ ?_ ?_ hfg <;> simp [hst]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- Specialization of `Finset.prod_bij` that automatically fills in most arguments.

See `Fintype.prod_bijective` for the version where `s` and `t` are `univ`. -/
@[to_additive "`Specialization of `Finset.sum_bij` that automatically fills in most arguments.

See `Fintype.sum_bijective` for the version where `s` and `t` are `univ`."]
lemma prod_bijective (e : ι → κ) (he : e.Bijective) (hst : ∀ i, i ∈ s ↔ e i ∈ t)
    (hfg : ∀ i ∈ s, f i = g (e i)) :
    ∏ i ∈ s, f i = ∏ i ∈ t, g i := prod_equiv (.ofBijective e he) hst hfg


@[to_additive]
lemma prod_of_injOn (e : ι → κ) (he : Set.InjOn e s) (hest : Set.MapsTo e s t)
    (h' : ∀ i ∈ t, i ∉ e '' s → g i = 1) (h : ∀ i ∈ s, f i = g (e i))  :
    ∏ i ∈ s, f i = ∏ j ∈ t, g j := by
  classical
  exact (prod_nbij e (fun a ↦ mem_image_of_mem e) he (by simp [Set.surjOn_image]) h).trans <|
    prod_subset (image_subset_iff.2 hest) <| by simpa using h'


@[to_additive]
lemma prod_fiberwise_eq_prod_filter (s : Finset ι) (t : Finset κ) (g : ι → κ) (f : ι → α) :
    ∏ j ∈ t, ∏ i ∈ s with g i = j, f i = ∏ i ∈ s with g i ∈ t, f i := by
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    s : Finset ι
    t : Finset κ
    g : ι → κ
    f : ι → α
    ⊢ Eq (t.prod fun j => (Finset.filter (fun i => Eq (g i) j) s).prod fun i => f  …
  -/
  rw [← prod_disjiUnion, disjiUnion_filter_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_fiberwise_eq_prod_filter' (s : Finset ι) (t : Finset κ) (g : ι → κ) (f : κ → α) :
    ∏ j ∈ t, ∏ i ∈ s with g i = j, f j = ∏ i ∈ s with g i ∈ t, f (g i) := by
  calc
    _ = ∏ j ∈ t, ∏ i ∈ s with g i = j, f (g i) :=
        prod_congr rfl fun j _ ↦ prod_congr rfl fun i hi ↦ by rw [(mem_filter.1 hi).2]
    _ = _ := prod_fiberwise_eq_prod_filter _ _ _ _


@[to_additive]
lemma prod_fiberwise_of_maps_to {g : ι → κ} (h : ∀ i ∈ s, g i ∈ t) (f : ι → α) :
    ∏ j ∈ t, ∏ i ∈ s with g i = j, f i = ∏ i ∈ s, f i := by
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝¹ : CommMonoid α
    s : Finset ι
    t : Finset κ
    inst✝ : DecidableEq κ
    g : ι → κ
    h : ∀ (i : ι), Membership.mem s i → Membership.mem t (g i)
    f : ι → α
    ⊢ Eq (t.prod fun j => (Finset.filter (fun i => Eq (g i) j) s).prod fun i => f  …
  -/
  rw [← prod_disjiUnion, disjiUnion_filter_eq_of_maps_to h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_fiberwise_of_maps_to' {g : ι → κ} (h : ∀ i ∈ s, g i ∈ t) (f : κ → α) :
    ∏ j ∈ t, ∏ i ∈ s with g i = j, f j = ∏ i ∈ s, f (g i) := by
  calc
    _ = ∏ j ∈ t, ∏ i ∈ s with g i = j, f (g i) :=
        prod_congr rfl fun y _ ↦ prod_congr rfl fun x hx ↦ by rw [(mem_filter.1 hx).2]
    _ = _ := prod_fiberwise_of_maps_to h _


@[to_additive]
lemma prod_fiberwise (s : Finset ι) (g : ι → κ) (f : ι → α) :
    ∏ j, ∏ i ∈ s with g i = j, f i = ∏ i ∈ s, f i :=
  prod_fiberwise_of_maps_to (fun _ _ ↦ mem_univ _) _


@[to_additive]
lemma prod_fiberwise' (s : Finset ι) (g : ι → κ) (f : κ → α) :
    ∏ j, ∏ i ∈ s with g i = j, f j = ∏ i ∈ s, f (g i) :=
  prod_fiberwise_of_maps_to' (fun _ _ ↦ mem_univ _) _


/-- Taking a product over `univ.pi t` is the same as taking the product over `Fintype.piFinset t`.
`univ.pi t` and `Fintype.piFinset t` are essentially the same `Finset`, but differ
in the type of their element, `univ.pi t` is a `Finset (Π a ∈ univ, t a)` and
`Fintype.piFinset t` is a `Finset (Π a, t a)`. -/
@[to_additive "Taking a sum over `univ.pi t` is the same as taking the sum over
`Fintype.piFinset t`. `univ.pi t` and `Fintype.piFinset t` are essentially the same `Finset`,
but differ in the type of their element, `univ.pi t` is a `Finset (Π a ∈ univ, t a)` and
`Fintype.piFinset t` is a `Finset (Π a, t a)`."]
lemma prod_univ_pi [DecidableEq ι] [Fintype ι] {κ : ι → Type*} (t : ∀ i, Finset (κ i))
    (f : (∀ i ∈ (univ : Finset ι), κ i) → β) :
    ∏ x ∈ univ.pi t, f x = ∏ x ∈ Fintype.piFinset t, f fun a _ ↦ x a := by
  /-
    ι : Type u_1
    β : Type u_4
    inst✝² : CommMonoid β
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    κ : ι → Type u_6
    t : (i : ι) → Finset (κ i)
    f : ((i : ι) → Membership.mem Finset.univ i → κ i) → β
    ⊢ Eq ((Finset.univ.pi t).prod fun x => f x) ((Fintype.piFinset t).prod fun x = …
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  apply prod_nbij' (fun x i ↦ x i <| mem_univ _) (fun x i _ ↦ x i) <;> simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive (attr := simp)]
lemma prod_diag [DecidableEq α] (s : Finset α) (f : α × α → β) :
    ∏ i ∈ s.diag, f i = ∏ i ∈ s, f (i, i) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    f : Prod α α → β
    ⊢ Eq (s.diag.prod fun i => f i) (s.prod fun i => f { fst := i, snd := i })
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  apply prod_nbij' Prod.fst (fun i ↦ (i, i)) <;> simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive]
theorem prod_finset_product (r : Finset (γ × α)) (s : Finset γ) (t : γ → Finset α)
    (h : ∀ p : γ × α, p ∈ r ↔ p.1 ∈ s ∧ p.2 ∈ t p.1) {f : γ × α → β} :
    ∏ p ∈ r, f p = ∏ c ∈ s, ∏ a ∈ t c, f (c, a) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    r : Finset (Prod γ α)
    s : Finset γ
    t : γ → Finset α
    h : ∀ (p : Prod γ α), Iff (Membership.mem r p) (And (Membership.mem s p.1) (Me …
    f : Prod γ α → β
    ⊢ Eq (r.prod fun p => f p) (s.prod fun c => (t c).prod fun a => f { fst := c,  …
  -/
  refine Eq.trans ?_ (prod_sigma s t fun p => f (p.1, p.2))
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    r : Finset (Prod γ α)
    s : Finset γ
    t : γ → Finset α
    h : ∀ (p : Prod γ α), Iff (Membership.mem r p) (And (Membership.mem s p.1) (Me …
    f : Prod γ α → β
    ⊢ Eq (r.prod fun p => f p) ((s.sigma t).prod fun x => f { fst := x.fst, snd := …
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  apply prod_equiv (Equiv.sigmaEquivProd _ _).symm <;> simp [h]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive]
theorem prod_finset_product' (r : Finset (γ × α)) (s : Finset γ) (t : γ → Finset α)
    (h : ∀ p : γ × α, p ∈ r ↔ p.1 ∈ s ∧ p.2 ∈ t p.1) {f : γ → α → β} :
    ∏ p ∈ r, f p.1 p.2 = ∏ c ∈ s, ∏ a ∈ t c, f c a :=
  prod_finset_product r s t h


@[to_additive]
theorem prod_finset_product_right (r : Finset (α × γ)) (s : Finset γ) (t : γ → Finset α)
    (h : ∀ p : α × γ, p ∈ r ↔ p.2 ∈ s ∧ p.1 ∈ t p.2) {f : α × γ → β} :
    ∏ p ∈ r, f p = ∏ c ∈ s, ∏ a ∈ t c, f (a, c) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    r : Finset (Prod α γ)
    s : Finset γ
    t : γ → Finset α
    h : ∀ (p : Prod α γ), Iff (Membership.mem r p) (And (Membership.mem s p.2) (Me …
    f : Prod α γ → β
    ⊢ Eq (r.prod fun p => f p) (s.prod fun c => (t c).prod fun a => f { fst := a,  …
  -/
  refine Eq.trans ?_ (prod_sigma s t fun p => f (p.2, p.1))
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝ : CommMonoid β
    r : Finset (Prod α γ)
    s : Finset γ
    t : γ → Finset α
    h : ∀ (p : Prod α γ), Iff (Membership.mem r p) (And (Membership.mem s p.2) (Me …
    f : Prod α γ → β
    ⊢ Eq (r.prod fun p => f p) ((s.sigma t).prod fun x => f { fst := x.snd, snd := …
  -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  apply prod_equiv ((Equiv.prodComm _ _).trans (Equiv.sigmaEquivProd _ _).symm) <;> simp [h]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[to_additive]
theorem prod_finset_product_right' (r : Finset (α × γ)) (s : Finset γ) (t : γ → Finset α)
    (h : ∀ p : α × γ, p ∈ r ↔ p.2 ∈ s ∧ p.1 ∈ t p.2) {f : α → γ → β} :
    ∏ p ∈ r, f p.1 p.2 = ∏ c ∈ s, ∏ a ∈ t c, f a c :=
  prod_finset_product_right r s t h


@[to_additive]
theorem prod_image' [DecidableEq α] {s : Finset ι} {g : ι → α} (h : ι → β)
    (eq : ∀ i ∈ s, f (g i) = ∏ j ∈ s with g j = g i, h j) :
    ∏ a ∈ s.image g, f a = ∏ i ∈ s, h i :=
  calc
    ∏ a ∈ s.image g, f a = ∏ a ∈ s.image g, ∏ j ∈ s with g j = a, h j :=
      (prod_congr rfl) fun _a hx =>
        let ⟨i, his, hi⟩ := mem_image.1 hx
        hi ▸ eq i his
    _ = ∏ i ∈ s, h i := prod_fiberwise_of_maps_to (fun _ => mem_image_of_mem g) _


@[to_additive]
theorem prod_mul_distrib : ∏ x ∈ s, f x * g x = (∏ x ∈ s, f x) * ∏ x ∈ s, g x :=
               /-
                 α : Type u_3
                 β : Type u_4
                 s : Finset α
                 f g : α → β
                 inst✝ : CommMonoid β
                 ⊢ Eq (s.prod fun x => HMul.hMul (f x) (g x)) (Finset.fold (fun x1 x2 => HMul.h …
               -/
  Eq.trans (by rw [one_mul]; rfl) fold_op_distrib
                             /-
                               🎉 no goals
                             -/


@[to_additive]
lemma prod_mul_prod_comm (f g h i : α → β) :
    (∏ a ∈ s, f a * g a) * ∏ a ∈ s, h a * i a = (∏ a ∈ s, f a * h a) * ∏ a ∈ s, g a * i a := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝ : CommMonoid β
    f g h i : α → β
    ⊢ Eq (HMul.hMul (s.prod fun a => HMul.hMul (f a) (g a)) (s.prod fun a => HMul. …
  -/
  simp_rw [prod_mul_distrib, mul_mul_mul_comm]
  /-
    🎉 no goals
  -/


/-- The product over a product set equals the product of the fiberwise products. For rewriting
in the reverse direction, use `Finset.prod_product'`. -/
@[to_additive "The sum over a product set equals the sum of the fiberwise sums. For rewriting
in the reverse direction, use `Finset.sum_product'`"]
theorem prod_product (s : Finset γ) (t : Finset α) (f : γ × α → β) :
    ∏ x ∈ s ×ˢ t, f x = ∏ x ∈ s, ∏ y ∈ t, f (x, y) :=
  prod_finset_product (s ×ˢ t) s (fun _a => t) fun _p => mem_product


/-- The product over a product set equals the product of the fiberwise products. For rewriting
in the reverse direction, use `Finset.prod_product`. -/
@[to_additive "The sum over a product set equals the sum of the fiberwise sums. For rewriting
in the reverse direction, use `Finset.sum_product`"]
theorem prod_product' (s : Finset γ) (t : Finset α) (f : γ → α → β) :
    ∏ x ∈ s ×ˢ t, f x.1 x.2 = ∏ x ∈ s, ∏ y ∈ t, f x y :=
  prod_product ..


@[to_additive]
theorem prod_product_right (s : Finset γ) (t : Finset α) (f : γ × α → β) :
    ∏ x ∈ s ×ˢ t, f x = ∏ y ∈ t, ∏ x ∈ s, f (x, y) :=
  prod_finset_product_right (s ×ˢ t) t (fun _a => s) fun _p => mem_product.trans and_comm


/-- An uncurried version of `Finset.prod_product_right`. -/
@[to_additive "An uncurried version of `Finset.sum_product_right`"]
theorem prod_product_right' (s : Finset γ) (t : Finset α) (f : γ → α → β) :
    ∏ x ∈ s ×ˢ t, f x.1 x.2 = ∏ y ∈ t, ∏ x ∈ s, f x y :=
  prod_product_right ..


/-- Generalization of `Finset.prod_comm` to the case when the inner `Finset`s depend on the outer
variable. -/
@[to_additive "Generalization of `Finset.sum_comm` to the case when the inner `Finset`s depend on
the outer variable."]
theorem prod_comm' {s : Finset γ} {t : γ → Finset α} {t' : Finset α} {s' : α → Finset γ}
    (h : ∀ x y, x ∈ s ∧ y ∈ t x ↔ x ∈ s' y ∧ y ∈ t') {f : γ → α → β} :
    (∏ x ∈ s, ∏ y ∈ t x, f x y) = ∏ y ∈ t', ∏ x ∈ s' y, f x y := by
  classical
    have : ∀ z : γ × α, (z ∈ s.biUnion fun x => (t x).map <| Function.Embedding.sectR x _) ↔
      z.1 ∈ s ∧ z.2 ∈ t z.1 := by
      rintro ⟨x, y⟩
      simp only [mem_biUnion, mem_map, Function.Embedding.sectR_apply, Prod.mk.injEq,
        exists_eq_right, ← and_assoc]
    exact
      (prod_finset_product' _ _ _ this).symm.trans
        ((prod_finset_product_right' _ _ _) fun ⟨x, y⟩ => (this _).trans ((h x y).trans and_comm))


@[to_additive]
theorem prod_comm {s : Finset γ} {t : Finset α} {f : γ → α → β} :
    (∏ x ∈ s, ∏ y ∈ t, f x y) = ∏ y ∈ t, ∏ x ∈ s, f x y :=
  prod_comm' fun _ _ => Iff.rfl


@[to_additive]
theorem prod_hom_rel [CommMonoid γ] {r : β → γ → Prop} {f : α → β} {g : α → γ} {s : Finset α}
    (h₁ : r 1 1) (h₂ : ∀ a b c, r b c → r (f a * b) (g a * c)) :
    r (∏ x ∈ s, f x) (∏ x ∈ s, g x) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝¹ : CommMonoid β
    inst✝ : CommMonoid γ
    r : β → γ → Prop
    f : α → β
    g : α → γ
    s : Finset α
    h₁ : r 1 1
    h₂ : ∀ (a : α) (b : β) (c : γ), r b c → r (HMul.hMul (f a) b) (HMul.hMul (g a) …
    ⊢ r (s.prod fun x => f x) (s.prod fun x => g x)
  -/
  delta Finset.prod
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝¹ : CommMonoid β
    inst✝ : CommMonoid γ
    r : β → γ → Prop
    f : α → β
    g : α → γ
    s : Finset α
    h₁ : r 1 1
    h₂ : ∀ (a : α) (b : β) (c : γ), r b c → r (HMul.hMul (f a) b) (HMul.hMul (g a) …
    ⊢ r (Multiset.map (fun x => f x) s.val).prod (Multiset.map (fun x => g x) s.va …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  apply Multiset.prod_hom_rel <;> assumption
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive]
theorem prod_filter_of_ne {p : α → Prop} [DecidablePred p] (hp : ∀ x ∈ s, f x ≠ 1 → p x) :
    ∏ x ∈ s with p x, f x = ∏ x ∈ s, f x :=
  (prod_subset (filter_subset _ _)) fun x => by
    classical
      rw [not_imp_comm, mem_filter]
      exact fun h₁ h₂ => ⟨h₁, by simpa using hp _ h₁ h₂⟩

-- If we use `[DecidableEq β]` here, some rewrites fail because they find a wrong `Decidable`
-- instance first; `{∀ x, Decidable (f x ≠ 1)}` doesn't work with `rw ← prod_filter_ne_one`

@[to_additive]
theorem prod_filter_ne_one (s : Finset α) [∀ x, Decidable (f x ≠ 1)] :
    ∏ x ∈ s with f x ≠ 1, f x = ∏ x ∈ s, f x :=
  prod_filter_of_ne fun _ _ => id


@[to_additive]
theorem prod_filter (p : α → Prop) [DecidablePred p] (f : α → β) :
    ∏ a ∈ s with p a, f a = ∏ a ∈ s, if p a then f a else 1 :=
  calc
    ∏ a ∈ s with p a, f a = ∏ a ∈ s with p a, if p a then f a else 1 :=
                                   /-
                                     α : Type u_3
                                     β : Type u_4
                                     s : Finset α
                                     inst✝¹ : CommMonoid β
                                     p : α → Prop
                                     inst✝ : DecidablePred p
                                     f : α → β
                                     a : α
                                     h : Membership.mem (Finset.filter (fun a => p a) s) a
                                     ⊢ Eq (f a) (ite (p a) (f a) 1)
                                   -/
      prod_congr rfl fun a h => by rw [if_pos]; simpa using (mem_filter.1 h).2
                                                /-
                                                  🎉 no goals
                                                -/
    _ = ∏ a ∈ s, if p a then f a else 1 := by
      { refine prod_subset (filter_subset _ s) fun x hs h => ?_
        rw [mem_filter, not_and] at h
        exact if_neg (by simpa using h hs) }


@[to_additive]
theorem prod_eq_single_of_mem {s : Finset α} {f : α → β} (a : α) (h : a ∈ s)
    (h₀ : ∀ b ∈ s, b ≠ a → f b = 1) : ∏ x ∈ s, f x = f a := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    a : α
    h : Membership.mem s a
    h₀ : ∀ (b : α), Membership.mem s b → Ne b a → Eq (f b) 1
    ⊢ Eq (s.prod fun x => f x) (f a)
  -/
  haveI := Classical.decEq α
  calc
    ∏ x ∈ s, f x = ∏ x ∈ {a}, f x := by
      { refine (prod_subset ?_ ?_).symm
        · intro _ H
          rwa [mem_singleton.1 H]
        · simpa only [mem_singleton] }
    _ = f a := prod_singleton _ _


@[to_additive]
theorem prod_eq_single {s : Finset α} {f : α → β} (a : α) (h₀ : ∀ b ∈ s, b ≠ a → f b = 1)
    (h₁ : a ∉ s → f a = 1) : ∏ x ∈ s, f x = f a :=
  haveI := Classical.decEq α
  by_cases (prod_eq_single_of_mem a · h₀) fun this =>
                                              /-
                                                α : Type u_3
                                                β : Type u_4
                                                inst✝ : CommMonoid β
                                                s : Finset α
                                                f : α → β
                                                a : α
                                                h₀ : ∀ (b : α), Membership.mem s b → Ne b a → Eq (f b) 1
                                                h₁ : Not (Membership.mem s a) → Eq (f a) 1
                                                this✝ : DecidableEq α
                                                this : Not (Membership.mem s a)
                                                b : α
                                                hb : Membership.mem s b
                                                ⊢ Ne b a
                                              -/
    (prod_congr rfl fun b hb => h₀ b hb <| by rintro rfl; exact this hb).trans <|
                                                          /-
                                                            🎉 no goals
                                                          -/
      prod_const_one.trans (h₁ this).symm


@[to_additive]
lemma prod_union_eq_left [DecidableEq α] (hs : ∀ a ∈ s₂, a ∉ s₁ → f a = 1) :
    ∏ a ∈ s₁ ∪ s₂, f a = ∏ a ∈ s₁, f a :=
  Eq.symm <|
    prod_subset subset_union_left fun _a ha ha' ↦ hs _ ((mem_union.1 ha).resolve_left ha') ha'


@[to_additive]
lemma prod_union_eq_right [DecidableEq α] (hs : ∀ a ∈ s₁, a ∉ s₂ → f a = 1) :
                                             /-
                                               α : Type u_3
                                               β : Type u_4
                                               s₁ s₂ : Finset α
                                               f : α → β
                                               inst✝¹ : CommMonoid β
                                               inst✝ : DecidableEq α
                                               hs : ∀ (a : α), Membership.mem s₁ a → Not (Membership.mem s₂ a) → Eq (f a) 1
                                               ⊢ Eq ((Union.union s₁ s₂).prod fun a => f a) (s₂.prod fun a => f a)
                                             -/
    ∏ a ∈ s₁ ∪ s₂, f a = ∏ a ∈ s₂, f a := by rw [union_comm, prod_union_eq_left hs]
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive]
theorem prod_eq_mul_of_mem {s : Finset α} {f : α → β} (a b : α) (ha : a ∈ s) (hb : b ∈ s)
    (hn : a ≠ b) (h₀ : ∀ c ∈ s, c ≠ a ∧ c ≠ b → f c = 1) : ∏ x ∈ s, f x = f a * f b := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    hn : Ne a b
    h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
    ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
  -/
  haveI := Classical.decEq α; let s' := ({a, b} : Finset α)
  have hu : s' ⊆ s := by
    refine insert_subset_iff.mpr ?_
    apply And.intro ha
    apply singleton_subset_iff.mpr hb
  have hf : ∀ c ∈ s, c ∉ s' → f c = 1 := by
    intro c hc hcs
    apply h₀ c hc
    apply not_or.mp
    intro hab
    apply hcs
    rw [mem_insert, mem_singleton]
    exact hab
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    hn : Ne a b
    h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
    this : DecidableEq α
    s' : Finset α := Insert.insert a (Singleton.singleton b)
    hu : HasSubset.Subset s' s
    hf : ∀ (c : α), Membership.mem s c → Not (Membership.mem s' c) → Eq (f c) 1
    ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
  -/
  rw [← prod_subset hu hf]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    hn : Ne a b
    h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
    this : DecidableEq α
    s' : Finset α := Insert.insert a (Singleton.singleton b)
    hu : HasSubset.Subset s' s
    hf : ∀ (c : α), Membership.mem s c → Not (Membership.mem s' c) → Eq (f c) 1
    ⊢ Eq (s'.prod fun x => f x) (HMul.hMul (f a) (f b))
  -/
  exact Finset.prod_pair hn
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_eq_mul {s : Finset α} {f : α → β} (a b : α) (hn : a ≠ b)
    (h₀ : ∀ c ∈ s, c ≠ a ∧ c ≠ b → f c = 1) (ha : a ∉ s → f a = 1) (hb : b ∉ s → f b = 1) :
    ∏ x ∈ s, f x = f a * f b := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    a b : α
    hn : Ne a b
    h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
    ha : Not (Membership.mem s a) → Eq (f a) 1
    hb : Not (Membership.mem s b) → Eq (f b) 1
    ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
  -/
  haveI := Classical.decEq α; by_cases h₁ : a ∈ s <;> by_cases h₂ : b ∈ s
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Membership.mem s a
      h₂ : Membership.mem s b
      ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
    -/
  · exact prod_eq_mul_of_mem a b h₁ h₂ hn h₀
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Membership.mem s a
      h₂ : Not (Membership.mem s b)
      ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
    -/
  · rw [hb h₂, mul_one]
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Membership.mem s a
      h₂ : Not (Membership.mem s b)
      ⊢ Eq (s.prod fun x => f x) (f a)
    -/
    apply prod_eq_single_of_mem a h₁
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Membership.mem s a
      h₂ : Not (Membership.mem s b)
      ⊢ ∀ (b : α), Membership.mem s b → Ne b a → Eq (f b) 1
    -/
    exact fun c hc hca => h₀ c hc ⟨hca, ne_of_mem_of_not_mem hc h₂⟩
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Not (Membership.mem s a)
      h₂ : Membership.mem s b
      ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
    -/
  · rw [ha h₁, one_mul]
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Not (Membership.mem s a)
      h₂ : Membership.mem s b
      ⊢ Eq (s.prod fun x => f x) (f b)
    -/
    apply prod_eq_single_of_mem b h₂
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Not (Membership.mem s a)
      h₂ : Membership.mem s b
      ⊢ ∀ (b_1 : α), Membership.mem s b_1 → Ne b_1 b → Eq (f b_1) 1
    -/
    exact fun c hc hcb => h₀ c hc ⟨ne_of_mem_of_not_mem hc h₁, hcb⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      f : α → β
      a b : α
      hn : Ne a b
      h₀ : ∀ (c : α), Membership.mem s c → And (Ne c a) (Ne c b) → Eq (f c) 1
      ha : Not (Membership.mem s a) → Eq (f a) 1
      hb : Not (Membership.mem s b) → Eq (f b) 1
      this : DecidableEq α
      h₁ : Not (Membership.mem s a)
      h₂ : Not (Membership.mem s b)
      ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f a) (f b))
    -/
  · rw [ha h₁, hb h₂, mul_one]
    exact
      _root_.trans
        (prod_congr rfl fun c hc =>
          h₀ c hc ⟨ne_of_mem_of_not_mem hc h₁, ne_of_mem_of_not_mem hc h₂⟩)
        prod_const_one


/-- A product over `s.subtype p` equals one over `{x ∈ s | p x}`. -/
@[to_additive (attr := simp)
"A sum over `s.subtype p` equals one over `{x ∈ s | p x}`."]
theorem prod_subtype_eq_prod_filter (f : α → β) {p : α → Prop} [DecidablePred p] :
    ∏ x ∈ s.subtype p, f x = ∏ x ∈ s with p x, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝¹ : CommMonoid β
    f : α → β
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq ((Finset.subtype p s).prod fun x => f ↑x) ((Finset.filter (fun x => p x)  …
  -/
  conv_lhs => erw [← prod_map (s.subtype p) (Function.Embedding.subtype _) f]
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝¹ : CommMonoid β
    f : α → β
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq ((Finset.map (Function.Embedding.subtype p) (Finset.subtype p s)).prod fu …
  -/
  exact prod_congr (subtype_map _) fun x _hx => rfl
  /-
    🎉 no goals
  -/


/-- If all elements of a `Finset` satisfy the predicate `p`, a product
over `s.subtype p` equals that product over `s`. -/
@[to_additive "If all elements of a `Finset` satisfy the predicate `p`, a sum
over `s.subtype p` equals that sum over `s`."]
theorem prod_subtype_of_mem (f : α → β) {p : α → Prop} [DecidablePred p] (h : ∀ x ∈ s, p x) :
    ∏ x ∈ s.subtype p, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝¹ : CommMonoid β
    f : α → β
    p : α → Prop
    inst✝ : DecidablePred p
    h : ∀ (x : α), Membership.mem s x → p x
    ⊢ Eq ((Finset.subtype p s).prod fun x => f ↑x) (s.prod fun x => f x)
  -/
  rw [prod_subtype_eq_prod_filter, filter_true_of_mem]
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝¹ : CommMonoid β
    f : α → β
    p : α → Prop
    inst✝ : DecidablePred p
    h : ∀ (x : α), Membership.mem s x → p x
    ⊢ ∀ (x : α), Membership.mem s x → p x
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/-- A product of a function over a `Finset` in a subtype equals a
product in the main type of a function that agrees with the first
function on that `Finset`. -/
@[to_additive "A sum of a function over a `Finset` in a subtype equals a
sum in the main type of a function that agrees with the first
function on that `Finset`."]
theorem prod_subtype_map_embedding {p : α → Prop} {s : Finset { x // p x }} {f : { x // p x } → β}
    {g : α → β} (h : ∀ x : { x // p x }, x ∈ s → g x = f x) :
    (∏ x ∈ s.map (Function.Embedding.subtype _), g x) = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    p : α → Prop
    s : Finset (Subtype fun x => p x)
    f : (Subtype fun x => p x) → β
    g : α → β
    h : ∀ (x : Subtype fun x => p x), Membership.mem s x → Eq (g ↑x) (f x)
    ⊢ Eq ((Finset.map (Function.Embedding.subtype fun x => p x) s).prod fun x => g …
  -/
  rw [Finset.prod_map]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    p : α → Prop
    s : Finset (Subtype fun x => p x)
    f : (Subtype fun x => p x) → β
    g : α → β
    h : ∀ (x : Subtype fun x => p x), Membership.mem s x → Eq (g ↑x) (f x)
    ⊢ Eq (s.prod fun x => g ((Function.Embedding.subtype fun x => p x) x)) (s.prod …
  -/
  exact Finset.prod_congr rfl h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_coe_sort_eq_attach (f : s → β) : ∏ i : s, f i = ∏ i ∈ s.attach, f i :=
  rfl


@[to_additive]
theorem prod_coe_sort : ∏ i : s, f i = ∏ i ∈ s, f i := prod_attach _ _


@[to_additive]
theorem prod_finset_coe (f : α → β) (s : Finset α) : (∏ i : (s : Set α), f i) = ∏ i ∈ s, f i :=
  prod_coe_sort s f


@[to_additive]
theorem prod_subtype {p : α → Prop} {F : Fintype (Subtype p)} (s : Finset α) (h : ∀ x, x ∈ s ↔ p x)
    (f : α → β) : ∏ a ∈ s, f a = ∏ a : Subtype p, f a := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    p : α → Prop
    F : Fintype (Subtype p)
    s : Finset α
    h : ∀ (x : α), Iff (Membership.mem s x) (p x)
    f : α → β
    ⊢ Eq (s.prod fun a => f a) (Finset.univ.prod fun a => f ↑a)
  -/
  have : (· ∈ s) = p := Set.ext h
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    p : α → Prop
    F : Fintype (Subtype p)
    s : Finset α
    h : ∀ (x : α), Iff (Membership.mem s x) (p x)
    f : α → β
    this : Eq (fun x => Membership.mem s x) p
    ⊢ Eq (s.prod fun a => f a) (Finset.univ.prod fun a => f ↑a)
  -/
  subst p
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    F : Fintype (Subtype fun x => Membership.mem s x)
    h : ∀ (x : α), Iff (Membership.mem s x) ((fun x => Membership.mem s x) x)
    ⊢ Eq (s.prod fun a => f a) (Finset.univ.prod fun a => f ↑a)
  -/
  rw [← prod_coe_sort]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    F : Fintype (Subtype fun x => Membership.mem s x)
    h : ∀ (x : α), Iff (Membership.mem s x) ((fun x => Membership.mem s x) x)
    ⊢ Eq (Finset.univ.prod fun i => f ↑i) (Finset.univ.prod fun a => f ↑a)
  -/
  congr!
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_preimage' (f : ι → κ) [DecidablePred (· ∈ Set.range f)] (s : Finset κ) (hf) (g : κ → β) :
    ∏ x ∈ s.preimage f hf, g (f x) = ∏ x ∈ s with x ∈ Set.range f, g x := by
  classical
  calc
    ∏ x ∈ preimage s f hf, g (f x) = ∏ x ∈ image f (preimage s f hf), g x :=
      Eq.symm <| prod_image <| by simpa only [mem_preimage, Set.InjOn] using hf
    _ = ∏ x ∈ s with x ∈ Set.range f, g x := by rw [image_preimage]


@[to_additive]
lemma prod_preimage (f : ι → κ) (s : Finset κ) (hf) (g : κ → β)
    (hg : ∀ x ∈ s, x ∉ Set.range f → g x = 1) :
    ∏ x ∈ s.preimage f hf, g (f x) = ∏ x ∈ s, g x := by
  /-
    ι : Type u_1
    κ : Type u_2
    β : Type u_4
    inst✝ : CommMonoid β
    f : ι → κ
    s : Finset κ
    hf : Set.InjOn f (Set.preimage f ↑s)
    g : κ → β
    hg : ∀ (x : κ), Membership.mem s x → Not (Membership.mem (Set.range f) x) → Eq …
    ⊢ Eq ((s.preimage f hf).prod fun x => g (f x)) (s.prod fun x => g x)
  -/
  classical rw [prod_preimage', prod_filter_of_ne]; exact fun x hx ↦ Not.imp_symm (hg x hx)
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_preimage_of_bij (f : ι → κ) (s : Finset κ) (hf : Set.BijOn f (f ⁻¹' ↑s) ↑s) (g : κ → β) :
    ∏ x ∈ s.preimage f hf.injOn, g (f x) = ∏ x ∈ s, g x :=
  prod_preimage _ _ hf.injOn g fun _ hs h_f ↦ (h_f <| hf.subset_range hs).elim


@[to_additive]
theorem prod_set_coe (s : Set α) [Fintype s] : (∏ i : s, f i) = ∏ i ∈ s.toFinset, f i :=
(Finset.prod_subtype s.toFinset (fun _ ↦ Set.mem_toFinset) f).symm


/-- The product of a function `g` defined only on a set `s` is equal to
the product of a function `f` defined everywhere,
as long as `f` and `g` agree on `s`, and `f = 1` off `s`. -/
@[to_additive "The sum of a function `g` defined only on a set `s` is equal to
the sum of a function `f` defined everywhere,
as long as `f` and `g` agree on `s`, and `f = 0` off `s`."]
theorem prod_congr_set {α : Type*} [CommMonoid α] {β : Type*} [Fintype β] (s : Set β)
    [DecidablePred (· ∈ s)] (f : β → α) (g : s → α) (w : ∀ (x : β) (h : x ∈ s), f x = g ⟨x, h⟩)
    (w' : ∀ x : β, x ∉ s → f x = 1) : Finset.univ.prod f = Finset.univ.prod g := by
  /-
    α : Type u_6
    inst✝² : CommMonoid α
    β : Type u_7
    inst✝¹ : Fintype β
    s : Set β
    inst✝ : DecidablePred fun x => Membership.mem s x
    f : β → α
    g : ↑s → α
    w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
    w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
    ⊢ Eq (Finset.univ.prod f) (Finset.univ.prod g)
  -/
  rw [← @Finset.prod_subset _ _ s.toFinset Finset.univ f _ (by simp)]
    /-
      α : Type u_6
      inst✝² : CommMonoid α
      β : Type u_7
      inst✝¹ : Fintype β
      s : Set β
      inst✝ : DecidablePred fun x => Membership.mem s x
      f : β → α
      g : ↑s → α
      w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
      w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
      ⊢ Eq (s.toFinset.prod fun x => f x) (Finset.univ.prod g)
    -/
  · rw [Finset.prod_subtype]
      /-
        α : Type u_6
        inst✝² : CommMonoid α
        β : Type u_7
        inst✝¹ : Fintype β
        s : Set β
        inst✝ : DecidablePred fun x => Membership.mem s x
        f : β → α
        g : ↑s → α
        w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
        w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
        ⊢ Eq (Finset.univ.prod fun a => f ↑a) (Finset.univ.prod g)
      -/
    · apply Finset.prod_congr rfl
      /-
        α : Type u_6
        inst✝² : CommMonoid α
        β : Type u_7
        inst✝¹ : Fintype β
        s : Set β
        inst✝ : DecidablePred fun x => Membership.mem s x
        f : β → α
        g : ↑s → α
        w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
        w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
        ⊢ ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem Finset.univ x →  …
      -/
      exact fun ⟨x, h⟩ _ => w x h
      /-
        🎉 no goals
      -/
      /-
        case h
        α : Type u_6
        inst✝² : CommMonoid α
        β : Type u_7
        inst✝¹ : Fintype β
        s : Set β
        inst✝ : DecidablePred fun x => Membership.mem s x
        f : β → α
        g : ↑s → α
        w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
        w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
        ⊢ ∀ (x : β), Iff (Membership.mem s.toFinset x) (Membership.mem s x)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      α : Type u_6
      inst✝² : CommMonoid α
      β : Type u_7
      inst✝¹ : Fintype β
      s : Set β
      inst✝ : DecidablePred fun x => Membership.mem s x
      f : β → α
      g : ↑s → α
      w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
      w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
      ⊢ ∀ (x : β), Membership.mem Finset.univ x → Not (Membership.mem s.toFinset x)  …
    -/
  · rintro x _ h
    /-
      α : Type u_6
      inst✝² : CommMonoid α
      β : Type u_7
      inst✝¹ : Fintype β
      s : Set β
      inst✝ : DecidablePred fun x => Membership.mem s x
      f : β → α
      g : ↑s → α
      w : ∀ (x : β) (h : Membership.mem s x), Eq (f x) (g ⟨x, h⟩)
      w' : ∀ (x : β), Not (Membership.mem s x) → Eq (f x) 1
      x : β
      a✝ : Membership.mem Finset.univ x
      h : Not (Membership.mem s.toFinset x)
      ⊢ Eq (f x) 1
    -/
    exact w' x (by simpa using h)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_apply_dite {s : Finset α} {p : α → Prop} {hp : DecidablePred p}
    [DecidablePred fun x => ¬p x] (f : ∀ x : α, p x → γ) (g : ∀ x : α, ¬p x → γ) (h : γ → β) :
    (∏ x ∈ s, h (if hx : p x then f x hx else g x hx)) =
                                           /-
                                             ι : Type u_1
                                             κ : Type u_2
                                             α : Type u_3
                                             β : Type u_4
                                             γ : Type u_5
                                             s✝ s₁ s₂ : Finset α
                                             a : α
                                             f✝ g✝ : α → β
                                             inst✝¹ : CommMonoid β
                                             s : Finset α
                                             p : α → Prop
                                             hp : DecidablePred p
                                             inst✝ : DecidablePred fun x => Not (p x)
                                             f : (x : α) → p x → γ
                                             g : (x : α) → Not (p x) → γ
                                             h : γ → β
                                             x : Subtype fun x => Membership.mem (Finset.filter (fun x => p x) s) x
                                             ⊢ p ↑x
                                           -/
      (∏ x : {x ∈ s | p x}, h (f x.1 <| by simpa using (mem_filter.mp x.2).2)) *
                                           /-
                                             🎉 no goals
                                           -/
                                             /-
                                               ι : Type u_1
                                               κ : Type u_2
                                               α : Type u_3
                                               β : Type u_4
                                               γ : Type u_5
                                               s✝ s₁ s₂ : Finset α
                                               a : α
                                               f✝ g✝ : α → β
                                               inst✝¹ : CommMonoid β
                                               s : Finset α
                                               p : α → Prop
                                               hp : DecidablePred p
                                               inst✝ : DecidablePred fun x => Not (p x)
                                               f : (x : α) → p x → γ
                                               g : (x : α) → Not (p x) → γ
                                               h : γ → β
                                               x : Subtype fun x => Membership.mem (Finset.filter (fun x => Not (p x)) s) x
                                               ⊢ Not (p ↑x)
                                             -/
        ∏ x : {x ∈ s | ¬p x}, h (g x.1 <| by simpa using (mem_filter.mp x.2).2) :=
                                             /-
                                               🎉 no goals
                                             -/
  calc
    (∏ x ∈ s, h (if hx : p x then f x hx else g x hx)) =
        (∏ x ∈ s with p x, h (if hx : p x then f x hx else g x hx)) *
          ∏ x ∈ s with ¬p x, h (if hx : p x then f x hx else g x hx) :=
      (prod_filter_mul_prod_filter_not s p _).symm
    _ = (∏ x : {x ∈ s | p x}, h (if hx : p x.1 then f x.1 hx else g x.1 hx)) *
          ∏ x : {x ∈ s | ¬p x}, h (if hx : p x.1 then f x.1 hx else g x.1 hx) :=
      congr_arg₂ _ (prod_attach _ _).symm (prod_attach _ _).symm
                                             /-
                                               α : Type u_3
                                               β : Type u_4
                                               γ : Type u_5
                                               inst✝¹ : CommMonoid β
                                               s : Finset α
                                               p : α → Prop
                                               hp : DecidablePred p
                                               inst✝ : DecidablePred fun x => Not (p x)
                                               f : (x : α) → p x → γ
                                               g : (x : α) → Not (p x) → γ
                                               h : γ → β
                                               x : Subtype fun x => Membership.mem (Finset.filter (fun x => p x) s) x
                                               ⊢ p ↑x
                                             -/
    _ = (∏ x : {x ∈ s | p x}, h (f x.1 <| by simpa using (mem_filter.mp x.2).2)) *
                                             /-
                                               🎉 no goals
                                             -/
                                               /-
                                                 α : Type u_3
                                                 β : Type u_4
                                                 γ : Type u_5
                                                 inst✝¹ : CommMonoid β
                                                 s : Finset α
                                                 p : α → Prop
                                                 hp : DecidablePred p
                                                 inst✝ : DecidablePred fun x => Not (p x)
                                                 f : (x : α) → p x → γ
                                                 g : (x : α) → Not (p x) → γ
                                                 h : γ → β
                                                 x : Subtype fun x => Membership.mem (Finset.filter (fun x => Not (p x)) s) x
                                                 ⊢ Not (p ↑x)
                                               -/
          ∏ x : {x ∈ s | ¬p x}, h (g x.1 <| by simpa using (mem_filter.mp x.2).2) :=
                                               /-
                                                 🎉 no goals
                                               -/
      congr_arg₂ _ (prod_congr rfl fun x _hx ↦
                                   /-
                                     α : Type u_3
                                     β : Type u_4
                                     γ : Type u_5
                                     inst✝¹ : CommMonoid β
                                     s : Finset α
                                     p : α → Prop
                                     hp : DecidablePred p
                                     inst✝ : DecidablePred fun x => Not (p x)
                                     f : (x : α) → p x → γ
                                     g : (x : α) → Not (p x) → γ
                                     h : γ → β
                                     x : Subtype fun x => Membership.mem (Finset.filter (fun x => p x) s) x
                                     _hx : Membership.mem Finset.univ x
                                     ⊢ p ↑x
                                   -/
        congr_arg h (dif_pos <| by simpa using (mem_filter.mp x.2).2))
                                   /-
                                     🎉 no goals
                                   -/
                                                                /-
                                                                  α : Type u_3
                                                                  β : Type u_4
                                                                  γ : Type u_5
                                                                  inst✝¹ : CommMonoid β
                                                                  s : Finset α
                                                                  p : α → Prop
                                                                  hp : DecidablePred p
                                                                  inst✝ : DecidablePred fun x => Not (p x)
                                                                  f : (x : α) → p x → γ
                                                                  g : (x : α) → Not (p x) → γ
                                                                  h : γ → β
                                                                  x : Subtype fun x => Membership.mem (Finset.filter (fun x => Not (p x)) s) x
                                                                  _hx : Membership.mem Finset.univ x
                                                                  ⊢ Not (p ↑x)
                                                                -/
        (prod_congr rfl fun x _hx => congr_arg h (dif_neg <| by simpa using (mem_filter.mp x.2).2))
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive]
theorem prod_apply_ite {s : Finset α} {p : α → Prop} {_hp : DecidablePred p} (f g : α → γ)
    (h : γ → β) :
    (∏ x ∈ s, h (if p x then f x else g x)) =
      (∏ x ∈ s with p x, h (f x)) * ∏ x ∈ s with ¬p x, h (g x) :=
  (prod_apply_dite _ _ _).trans <| congr_arg₂ _ (prod_attach _ (h ∘ f)) (prod_attach _ (h ∘ g))


@[to_additive]
theorem prod_dite {s : Finset α} {p : α → Prop} {hp : DecidablePred p} (f : ∀ x : α, p x → β)
    (g : ∀ x : α, ¬p x → β) :
    ∏ x ∈ s, (if hx : p x then f x hx else g x hx) =
                                      /-
                                        ι : Type u_1
                                        κ : Type u_2
                                        α : Type u_3
                                        β : Type u_4
                                        γ : Type u_5
                                        s✝ s₁ s₂ : Finset α
                                        a : α
                                        f✝ g✝ : α → β
                                        inst✝ : CommMonoid β
                                        s : Finset α
                                        p : α → Prop
                                        hp : DecidablePred p
                                        f : (x : α) → p x → β
                                        g : (x : α) → Not (p x) → β
                                        x : Subtype fun x => Membership.mem (Finset.filter (fun x => p x) s) x
                                        ⊢ p ↑x
                                      -/
      (∏ x : {x ∈ s | p x}, f x.1 (by simpa using (mem_filter.mp x.2).2)) *
                                      /-
                                        🎉 no goals
                                      -/
                                        /-
                                          ι : Type u_1
                                          κ : Type u_2
                                          α : Type u_3
                                          β : Type u_4
                                          γ : Type u_5
                                          s✝ s₁ s₂ : Finset α
                                          a : α
                                          f✝ g✝ : α → β
                                          inst✝ : CommMonoid β
                                          s : Finset α
                                          p : α → Prop
                                          hp : DecidablePred p
                                          f : (x : α) → p x → β
                                          g : (x : α) → Not (p x) → β
                                          x : Subtype fun x => Membership.mem (Finset.filter (fun x => Not (p x)) s) x
                                          ⊢ Not (p ↑x)
                                        -/
        ∏ x : {x ∈ s | ¬p x}, g x.1 (by simpa using (mem_filter.mp x.2).2) := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    p : α → Prop
    hp : DecidablePred p
    f : (x : α) → p x → β
    g : (x : α) → Not (p x) → β
    ⊢ Eq (s.prod fun x => dite (p x) (fun hx => f x hx) fun hx => g x hx) (HMul.hM …
  -/
  simp [prod_apply_dite _ _ fun x => x]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_ite {s : Finset α} {p : α → Prop} {hp : DecidablePred p} (f g : α → β) :
    ∏ x ∈ s, (if p x then f x else g x) = (∏ x ∈ s with p x, f x) * ∏ x ∈ s with ¬p x, g x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    p : α → Prop
    hp : DecidablePred p
    f g : α → β
    ⊢ Eq (s.prod fun x => ite (p x) (f x) (g x)) (HMul.hMul ((Finset.filter (fun x …
  -/
  simp [prod_apply_ite _ _ fun x => x]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_dite_of_false {p : α → Prop} {_ : DecidablePred p} (h : ∀ i ∈ s, ¬ p i)
    (f : ∀ i, p i → β) (g : ∀ i, ¬ p i → β) :
    ∏ i ∈ s, (if hi : p i then f i hi else g i hi) = ∏ i : s, g i.1 (h _ i.2) := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝ : CommMonoid β
    p : α → Prop
    x✝ : DecidablePred p
    h : ∀ (i : α), Membership.mem s i → Not (p i)
    f : (i : α) → p i → β
    g : (i : α) → Not (p i) → β
    ⊢ Eq (s.prod fun i => dite (p i) (fun hi => f i hi) fun hi => g i hi) (Finset. …
  -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  refine prod_bij' (fun x hx => ⟨x, hx⟩) (fun x _ ↦ x) ?_ ?_ ?_ ?_ ?_ <;> aesop
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive]
lemma prod_ite_of_false {p : α → Prop} {_ : DecidablePred p} (h : ∀ x ∈ s, ¬p x) (f g : α → β) :
    ∏ x ∈ s, (if p x then f x else g x) = ∏ x ∈ s, g x :=
  (prod_dite_of_false h _ _).trans (prod_attach _ _)


@[to_additive]
lemma prod_dite_of_true {p : α → Prop} {_ : DecidablePred p} (h : ∀ i ∈ s, p i) (f : ∀ i, p i → β)
    (g : ∀ i, ¬ p i → β) :
    ∏ i ∈ s, (if hi : p i then f i hi else g i hi) = ∏ i : s, f i.1 (h _ i.2) := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    inst✝ : CommMonoid β
    p : α → Prop
    x✝ : DecidablePred p
    h : ∀ (i : α), Membership.mem s i → p i
    f : (i : α) → p i → β
    g : (i : α) → Not (p i) → β
    ⊢ Eq (s.prod fun i => dite (p i) (fun hi => f i hi) fun hi => g i hi) (Finset. …
  -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  refine prod_bij' (fun x hx => ⟨x, hx⟩) (fun x _ ↦ x) ?_ ?_ ?_ ?_ ?_ <;> aesop
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive]
lemma prod_ite_of_true {p : α → Prop} {_ : DecidablePred p} (h : ∀ x ∈ s, p x) (f g : α → β) :
    ∏ x ∈ s, (if p x then f x else g x) = ∏ x ∈ s, f x :=
  (prod_dite_of_true h _ _).trans (prod_attach _ _)


@[to_additive]
theorem prod_apply_ite_of_false {p : α → Prop} {hp : DecidablePred p} (f g : α → γ) (k : γ → β)
    (h : ∀ x ∈ s, ¬p x) : (∏ x ∈ s, k (if p x then f x else g x)) = ∏ x ∈ s, k (g x) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    s : Finset α
    inst✝ : CommMonoid β
    p : α → Prop
    hp : DecidablePred p
    f g : α → γ
    k : γ → β
    h : ∀ (x : α), Membership.mem s x → Not (p x)
    ⊢ Eq (s.prod fun x => k (ite (p x) (f x) (g x))) (s.prod fun x => k (g x))
  -/
  simp_rw [apply_ite k]
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    s : Finset α
    inst✝ : CommMonoid β
    p : α → Prop
    hp : DecidablePred p
    f g : α → γ
    k : γ → β
    h : ∀ (x : α), Membership.mem s x → Not (p x)
    ⊢ Eq (s.prod fun x => ite (p x) (k (f x)) (k (g x))) (s.prod fun x => k (g x))
  -/
  exact prod_ite_of_false h _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_apply_ite_of_true {p : α → Prop} {hp : DecidablePred p} (f g : α → γ) (k : γ → β)
    (h : ∀ x ∈ s, p x) : (∏ x ∈ s, k (if p x then f x else g x)) = ∏ x ∈ s, k (f x) := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    s : Finset α
    inst✝ : CommMonoid β
    p : α → Prop
    hp : DecidablePred p
    f g : α → γ
    k : γ → β
    h : ∀ (x : α), Membership.mem s x → p x
    ⊢ Eq (s.prod fun x => k (ite (p x) (f x) (g x))) (s.prod fun x => k (f x))
  -/
  simp_rw [apply_ite k]
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    s : Finset α
    inst✝ : CommMonoid β
    p : α → Prop
    hp : DecidablePred p
    f g : α → γ
    k : γ → β
    h : ∀ (x : α), Membership.mem s x → p x
    ⊢ Eq (s.prod fun x => ite (p x) (k (f x)) (k (g x))) (s.prod fun x => k (f x))
  -/
  exact prod_ite_of_true h _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_extend_by_one [DecidableEq α] (s : Finset α) (f : α → β) :
    ∏ i ∈ s, (if i ∈ s then f i else 1) = ∏ i ∈ s, f i :=
  (prod_congr rfl) fun _i hi => if_pos hi


@[to_additive (attr := simp)]
theorem prod_ite_mem [DecidableEq α] (s t : Finset α) (f : α → β) :
    ∏ i ∈ s, (if i ∈ t then f i else 1) = ∏ i ∈ s ∩ t, f i := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s t : Finset α
    f : α → β
    ⊢ Eq (s.prod fun i => ite (Membership.mem t i) (f i) 1) ((Inter.inter s t).pro …
  -/
  rw [← Finset.prod_filter, Finset.filter_mem_eq_inter]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_dite_eq [DecidableEq α] (s : Finset α) (a : α) (b : ∀ x : α, a = x → β) :
    ∏ x ∈ s, (if h : a = x then b x h else 1) = ite (a ∈ s) (b a rfl) 1 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    b : (x : α) → Eq a x → β
    ⊢ Eq (s.prod fun x => dite (Eq a x) (fun h => b x h) fun h => 1) (ite (Members …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq a x → β
      h : Membership.mem s a
      ⊢ Eq (s.prod fun x => dite (Eq a x) (fun h => b x h) fun h => 1) (b a ⋯)
    -/
  · rw [Finset.prod_eq_single a, dif_pos rfl]
      /-
        case pos.h₀
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq a x → β
        h : Membership.mem s a
        ⊢ ∀ (b_1 : α), Membership.mem s b_1 → Ne b_1 a → Eq (dite (Eq a b_1) (fun h => …
      -/
    · intros _ _ h
      /-
        case pos.h₀
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq a x → β
        h✝ : Membership.mem s a
        b✝ : α
        a✝ : Membership.mem s b✝
        h : Ne b✝ a
        ⊢ Eq (dite (Eq a b✝) (fun h => b b✝ h) fun h => 1) 1
      -/
      rw [dif_neg]
      /-
        case pos.h₀.hnc
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq a x → β
        h✝ : Membership.mem s a
        b✝ : α
        a✝ : Membership.mem s b✝
        h : Ne b✝ a
        ⊢ Not (Eq a b✝)
      -/
      exact h.symm
      /-
        🎉 no goals
      -/
      /-
        case pos.h₁
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq a x → β
        h : Membership.mem s a
        ⊢ Not (Membership.mem s a) → Eq (dite (Eq a a) (fun h => b a h) fun h => 1) 1
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq a x → β
      h : Not (Membership.mem s a)
      ⊢ Eq (s.prod fun x => dite (Eq a x) (fun h => b x h) fun h => 1) 1
    -/
  · rw [Finset.prod_eq_one]
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq a x → β
      h : Not (Membership.mem s a)
      ⊢ ∀ (x : α), Membership.mem s x → Eq (dite (Eq a x) (fun h => b x h) fun h =>  …
    -/
    intros
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq a x → β
      h : Not (Membership.mem s a)
      x✝ : α
      a✝ : Membership.mem s x✝
      ⊢ Eq (dite (Eq a x✝) (fun h => b x✝ h) fun h => 1) 1
    -/
    rw [dif_neg]
    /-
      case neg.hnc
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq a x → β
      h : Not (Membership.mem s a)
      x✝ : α
      a✝ : Membership.mem s x✝
      ⊢ Not (Eq a x✝)
    -/
    rintro rfl
    /-
      case neg.hnc
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq a x → β
      h : Not (Membership.mem s a)
      a✝ : Membership.mem s a
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem prod_dite_eq' [DecidableEq α] (s : Finset α) (a : α) (b : ∀ x : α, x = a → β) :
    ∏ x ∈ s, (if h : x = a then b x h else 1) = ite (a ∈ s) (b a rfl) 1 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    b : (x : α) → Eq x a → β
    ⊢ Eq (s.prod fun x => dite (Eq x a) (fun h => b x h) fun h => 1) (ite (Members …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq x a → β
      h : Membership.mem s a
      ⊢ Eq (s.prod fun x => dite (Eq x a) (fun h => b x h) fun h => 1) (b a ⋯)
    -/
  · rw [Finset.prod_eq_single a, dif_pos rfl]
      /-
        case pos.h₀
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq x a → β
        h : Membership.mem s a
        ⊢ ∀ (b_1 : α), Membership.mem s b_1 → Ne b_1 a → Eq (dite (Eq b_1 a) (fun h => …
      -/
    · intros _ _ h
      /-
        case pos.h₀
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq x a → β
        h✝ : Membership.mem s a
        b✝ : α
        a✝ : Membership.mem s b✝
        h : Ne b✝ a
        ⊢ Eq (dite (Eq b✝ a) (fun h => b b✝ h) fun h => 1) 1
      -/
      rw [dif_neg]
      /-
        case pos.h₀.hnc
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq x a → β
        h✝ : Membership.mem s a
        b✝ : α
        a✝ : Membership.mem s b✝
        h : Ne b✝ a
        ⊢ Not (Eq b✝ a)
      -/
      exact h
      /-
        🎉 no goals
      -/
      /-
        case pos.h₁
        α : Type u_3
        β : Type u_4
        inst✝¹ : CommMonoid β
        inst✝ : DecidableEq α
        s : Finset α
        a : α
        b : (x : α) → Eq x a → β
        h : Membership.mem s a
        ⊢ Not (Membership.mem s a) → Eq (dite (Eq a a) (fun h => b a h) fun h => 1) 1
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq x a → β
      h : Not (Membership.mem s a)
      ⊢ Eq (s.prod fun x => dite (Eq x a) (fun h => b x h) fun h => 1) 1
    -/
  · rw [Finset.prod_eq_one]
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq x a → β
      h : Not (Membership.mem s a)
      ⊢ ∀ (x : α), Membership.mem s x → Eq (dite (Eq x a) (fun h => b x h) fun h =>  …
    -/
    intros
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq x a → β
      h : Not (Membership.mem s a)
      x✝ : α
      a✝ : Membership.mem s x✝
      ⊢ Eq (dite (Eq x✝ a) (fun h => b x✝ h) fun h => 1) 1
    -/
    rw [dif_neg]
    /-
      case neg.hnc
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      b : (x : α) → Eq x a → β
      h : Not (Membership.mem s a)
      x✝ : α
      a✝ : Membership.mem s x✝
      ⊢ Not (Eq x✝ a)
    -/
    rintro rfl
    /-
      case neg.hnc
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      s : Finset α
      x✝ : α
      a✝ : Membership.mem s x✝
      b : (x : α) → Eq x x✝ → β
      h : Not (Membership.mem s x✝)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem prod_ite_eq [DecidableEq α] (s : Finset α) (a : α) (b : α → β) :
    (∏ x ∈ s, ite (a = x) (b x) 1) = ite (a ∈ s) (b a) 1 :=
  prod_dite_eq s a fun x _ => b x


/-- A product taken over a conditional whose condition is an equality test on the index and whose
alternative is `1` has value either the term at that index or `1`.

The difference with `Finset.prod_ite_eq` is that the arguments to `Eq` are swapped. -/
@[to_additive (attr := simp) "A sum taken over a conditional whose condition is an equality
test on the index and whose alternative is `0` has value either the term at that index or `0`.

The difference with `Finset.sum_ite_eq` is that the arguments to `Eq` are swapped."]
theorem prod_ite_eq' [DecidableEq α] (s : Finset α) (a : α) (b : α → β) :
    (∏ x ∈ s, ite (x = a) (b x) 1) = ite (a ∈ s) (b a) 1 :=
  prod_dite_eq' s a fun x _ => b x


@[to_additive]
theorem prod_ite_eq_of_mem [DecidableEq α] (s : Finset α) (a : α) (b : α → β) (h : a ∈ s) :
    (∏ x ∈ s, if a = x then b x else 1) = b a := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    b : α → β
    h : Membership.mem s a
    ⊢ Eq (s.prod fun x => ite (Eq a x) (b x) 1) (b a)
  -/
  simp only [prod_ite_eq, if_pos h]
  /-
    🎉 no goals
  -/


/-- The difference with `Finset.prod_ite_eq_of_mem` is that the arguments to `Eq` are swapped. -/
@[to_additive]
theorem prod_ite_eq_of_mem' [DecidableEq α] (s : Finset α) (a : α) (b : α → β) (h : a ∈ s) :
    (∏ x ∈ s, if x = a then b x else 1) = b a := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    b : α → β
    h : Membership.mem s a
    ⊢ Eq (s.prod fun x => ite (Eq x a) (b x) 1) (b a)
  -/
  simp only [prod_ite_eq', if_pos h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_ite_index (p : Prop) [Decidable p] (s t : Finset α) (f : α → β) :
    ∏ x ∈ if p then s else t, f x = if p then ∏ x ∈ s, f x else ∏ x ∈ t, f x :=
  apply_ite (fun s => ∏ x ∈ s, f x) _ _ _


@[to_additive (attr := simp)]
theorem prod_ite_irrel (p : Prop) [Decidable p] (s : Finset α) (f g : α → β) :
    ∏ x ∈ s, (if p then f x else g x) = if p then ∏ x ∈ s, f x else ∏ x ∈ s, g x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    p : Prop
    inst✝ : Decidable p
    s : Finset α
    f g : α → β
    ⊢ Eq (s.prod fun x => ite p (f x) (g x)) (ite p (s.prod fun x => f x) (s.prod  …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> rfl
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp)]
theorem prod_dite_irrel (p : Prop) [Decidable p] (s : Finset α) (f : p → α → β) (g : ¬p → α → β) :
    ∏ x ∈ s, (if h : p then f h x else g h x) =
      if h : p then ∏ x ∈ s, f h x else ∏ x ∈ s, g h x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    p : Prop
    inst✝ : Decidable p
    s : Finset α
    f : p → α → β
    g : Not p → α → β
    ⊢ Eq (s.prod fun x => dite p (fun h => f h x) fun h => g h x) (dite p (fun h = …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> rfl
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp)]
theorem prod_pi_mulSingle' [DecidableEq α] (a : α) (x : β) (s : Finset α) :
    ∏ a' ∈ s, Pi.mulSingle a x a' = if a ∈ s then x else 1 :=
  prod_dite_eq' _ _ _


@[to_additive (attr := simp)]
theorem prod_pi_mulSingle {β : α → Type*} [DecidableEq α] [∀ a, CommMonoid (β a)] (a : α)
    (f : ∀ a, β a) (s : Finset α) :
    (∏ a' ∈ s, Pi.mulSingle a' (f a') a) = if a ∈ s then f a else 1 :=
  prod_dite_eq _ _ _


@[to_additive]
lemma mulSupport_prod (s : Finset ι) (f : ι → α → β) :
    mulSupport (fun x ↦ ∏ i ∈ s, f i x) ⊆ ⋃ i ∈ s, mulSupport (f i) := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset ι
    f : ι → α → β
    ⊢ HasSubset.Subset (Function.mulSupport fun x => s.prod fun i => f i x) (Set.i …
  -/
  simp only [mulSupport_subset_iff', Set.mem_iUnion, not_exists, nmem_mulSupport]
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset ι
    f : ι → α → β
    ⊢ ∀ (x : α), (∀ (x_1 : ι), Membership.mem s x_1 → Eq (f x_1 x) 1) → Eq (s.prod …
  -/
  exact fun x ↦ prod_eq_one
  /-
    🎉 no goals
  -/


/-- Consider a product of `g i (f i)` over a finset.  Suppose `g` is a function such as
`n ↦ (· ^ n)`, which maps a second argument of `1` to `1`. Then if `f` is replaced by the
corresponding multiplicative indicator function, the finset may be replaced by a possibly larger
finset without changing the value of the product. -/
@[to_additive "Consider a sum of `g i (f i)` over a finset.  Suppose `g` is a function such as
`n ↦ (n • ·)`, which maps a second argument of `0` to `0` (or a weighted sum of `f i * h i` or
`f i • h i`, where `f` gives the weights that are multiplied by some other function `h`). Then if
`f` is replaced by the corresponding indicator function, the finset may be replaced by a possibly
larger finset without changing the value of the sum."]
lemma prod_mulIndicator_subset_of_eq_one [One α] (f : ι → α) (g : ι → α → β) {s t : Finset ι}
    (h : s ⊆ t) (hg : ∀ a, g a 1 = 1) :
    ∏ i ∈ t, g i (mulIndicator ↑s f i) = ∏ i ∈ s, g i (f i) := by
  calc
    _ = ∏ i ∈ s, g i (mulIndicator ↑s f i) := by rw [prod_subset h fun i _ hn ↦ by simp [hn, hg]]
    _ = _ := prod_congr rfl fun i hi ↦ congr_arg _ <| mulIndicator_of_mem hi f


/-- Taking the product of an indicator function over a possibly larger finset is the same as
taking the original function over the original finset. -/
@[to_additive "Summing an indicator function over a possibly larger `Finset` is the same as summing
  the original function over the original finset."]
lemma prod_mulIndicator_subset (f : ι → β) {s t : Finset ι} (h : s ⊆ t) :
    ∏ i ∈ t, mulIndicator (↑s) f i = ∏ i ∈ s, f i :=
  prod_mulIndicator_subset_of_eq_one _ (fun _ ↦ id) h fun _ ↦ rfl


@[to_additive]
lemma prod_mulIndicator_eq_prod_filter (s : Finset ι) (f : ι → κ → β) (t : ι → Set κ) (g : ι → κ)
    [DecidablePred fun i ↦ g i ∈ t i] :
    ∏ i ∈ s, mulIndicator (t i) (f i) (g i) = ∏ i ∈ s with g i ∈ t i, f i (g i) := by
  refine (prod_filter_mul_prod_filter_not s (fun i ↦ g i ∈ t i) _).symm.trans <|
     Eq.trans (congr_arg₂ (· * ·) ?_ ?_) (mul_one _)
    /-
      case refine_1
      ι : Type u_1
      β : Type u_4
      inst✝¹ : CommMonoid β
      κ : Type u_6
      s : Finset ι
      f : ι → κ → β
      t : ι → Set κ
      g : ι → κ
      inst✝ : DecidablePred fun i => Membership.mem (t i) (g i)
      ⊢ Eq ((Finset.filter (fun x => Membership.mem (t x) (g x)) s).prod fun x => (t …
    -/
  · exact prod_congr rfl fun x hx ↦ mulIndicator_of_mem (mem_filter.1 hx).2 _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      β : Type u_4
      inst✝¹ : CommMonoid β
      κ : Type u_6
      s : Finset ι
      f : ι → κ → β
      t : ι → Set κ
      g : ι → κ
      inst✝ : DecidablePred fun i => Membership.mem (t i) (g i)
      ⊢ Eq ((Finset.filter (fun x => Not (Membership.mem (t x) (g x))) s).prod fun x …
    -/
  · exact prod_eq_one fun x hx ↦ mulIndicator_of_not_mem (mem_filter.1 hx).2 _
    /-
      🎉 no goals
    -/


@[to_additive]
lemma prod_mulIndicator_eq_prod_inter [DecidableEq ι] (s t : Finset ι) (f : ι → β) :
    ∏ i ∈ s, (t : Set ι).mulIndicator f i = ∏ i ∈ s ∩ t, f i := by
  /-
    ι : Type u_1
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq ι
    s t : Finset ι
    f : ι → β
    ⊢ Eq (s.prod fun i => (↑t).mulIndicator f i) ((Inter.inter s t).prod fun i =>  …
  -/
  rw [← filter_mem_eq_inter, prod_mulIndicator_eq_prod_filter]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive]
lemma mulIndicator_prod (s : Finset ι) (t : Set κ) (f : ι → κ → β) :
    mulIndicator t (∏ i ∈ s, f i) = ∏ i ∈ s, mulIndicator t (f i) :=
  map_prod (mulIndicatorHom _ _) _ _


@[to_additive]
lemma mulIndicator_biUnion (s : Finset ι) (t : ι → Set κ) {f : κ → β}
    (hs : (s : Set ι).PairwiseDisjoint t) :
    mulIndicator (⋃ i ∈ s, t i) f = fun a ↦ ∏ i ∈ s, mulIndicator (t i) f a := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons i s hi ih =>
    ext j
    rw [coe_cons, Set.pairwiseDisjoint_insert_of_not_mem (Finset.mem_coe.not.2 hi)] at hs
    classical
    rw [prod_cons, cons_eq_insert, set_biUnion_insert, mulIndicator_union_of_not_mem_inter, ih hs.1]
    exact (Set.disjoint_iff.mp (Set.disjoint_iUnion₂_right.mpr hs.2) ·)


@[to_additive]
lemma mulIndicator_biUnion_apply (s : Finset ι) (t : ι → Set κ) {f : κ → β}
    (h : (s : Set ι).PairwiseDisjoint t) (x : κ) :
    mulIndicator (⋃ i ∈ s, t i) f x = ∏ i ∈ s, mulIndicator (t i) f x := by
  /-
    ι : Type u_1
    β : Type u_4
    inst✝ : CommMonoid β
    κ : Type u_7
    s : Finset ι
    t : ι → Set κ
    f : κ → β
    h : (↑s).PairwiseDisjoint t
    x : κ
    ⊢ Eq ((Set.iUnion fun i => Set.iUnion fun h => t i).mulIndicator f x) (s.prod  …
  -/
  rw [mulIndicator_biUnion s t h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_bij_ne_one {s : Finset α} {t : Finset γ} {f : α → β} {g : γ → β}
    (i : ∀ a ∈ s, f a ≠ 1 → γ) (hi : ∀ a h₁ h₂, i a h₁ h₂ ∈ t)
    (i_inj : ∀ a₁ h₁₁ h₁₂ a₂ h₂₁ h₂₂, i a₁ h₁₁ h₁₂ = i a₂ h₂₁ h₂₂ → a₁ = a₂)
    (i_surj : ∀ b ∈ t, g b ≠ 1 → ∃ a h₁ h₂, i a h₁ h₂ = b) (h : ∀ a h₁ h₂, f a = g (i a h₁ h₂)) :
    ∏ x ∈ s, f x = ∏ x ∈ t, g x := by
  classical
  calc
    ∏ x ∈ s, f x = ∏ x ∈ s with f x ≠ 1, f x := by rw [prod_filter_ne_one]
    _ = ∏ x ∈ t with g x ≠ 1, g x :=
      prod_bij (fun a ha => i a (mem_filter.mp ha).1 <| by simpa using (mem_filter.mp ha).2)
        ?_ ?_ ?_ ?_
    _ = ∏ x ∈ t, g x := prod_filter_ne_one _
  · intros a ha
    refine (mem_filter.mp ha).elim ?_
    intros h₁ h₂
    refine (mem_filter.mpr ⟨hi a h₁ _, ?_⟩)
    specialize h a h₁ fun H ↦ by rw [H] at h₂; simp at h₂
    rwa [← h]
  · intros a₁ ha₁ a₂ ha₂
    refine (mem_filter.mp ha₁).elim fun _ha₁₁ _ha₁₂ ↦ ?_
    refine (mem_filter.mp ha₂).elim fun _ha₂₁ _ha₂₂ ↦ ?_
    apply i_inj
  · intros b hb
    refine (mem_filter.mp hb).elim fun h₁ h₂ ↦ ?_
    obtain ⟨a, ha₁, ha₂, eq⟩ := i_surj b h₁ fun H ↦ by rw [H] at h₂; simp at h₂
    exact ⟨a, mem_filter.mpr ⟨ha₁, ha₂⟩, eq⟩
  · refine (fun a ha => (mem_filter.mp ha).elim fun h₁ h₂ ↦ ?_)
    exact h a h₁ fun H ↦ by rw [H] at h₂; simp at h₂


@[to_additive]
theorem nonempty_of_prod_ne_one (h : ∏ x ∈ s, f x ≠ 1) : s.Nonempty :=
  s.eq_empty_or_nonempty.elim (fun H => False.elim <| h <| H.symm ▸ prod_empty) id


@[to_additive]
theorem exists_ne_one_of_prod_ne_one (h : ∏ x ∈ s, f x ≠ 1) : ∃ a ∈ s, f a ≠ 1 := by
  classical
    rw [← prod_filter_ne_one] at h
    rcases nonempty_of_prod_ne_one h with ⟨x, hx⟩
    exact ⟨x, (mem_filter.1 hx).1, by simpa using (mem_filter.1 hx).2⟩


@[to_additive]
theorem prod_range_succ_comm (f : ℕ → β) (n : ℕ) :
    (∏ x ∈ range (n + 1), f x) = f n * ∏ x ∈ range n, f x := by
  /-
    β : Type u_4
    inst✝ : CommMonoid β
    f : Nat → β
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun x => f x) (HMul.hMul (f n) ((Fin …
  -/
  rw [range_succ, prod_insert not_mem_range_self]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_range_succ (f : ℕ → β) (n : ℕ) :
    (∏ x ∈ range (n + 1), f x) = (∏ x ∈ range n, f x) * f n := by
  /-
    β : Type u_4
    inst✝ : CommMonoid β
    f : Nat → β
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun x => f x) (HMul.hMul ((Finset.ra …
  -/
  simp only [mul_comm, prod_range_succ_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_range_succ' (f : ℕ → β) :
    ∀ n : ℕ, (∏ k ∈ range (n + 1), f k) = (∏ k ∈ range n, f (k + 1)) * f 0
  | 0 => prod_range_succ _ _
                /-
                  β : Type u_4
                  inst✝ : CommMonoid β
                  f : Nat → β
                  n : Nat
                  ⊢ Eq ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).prod fun k => f k) (HMul.hM …
                -/
  | n + 1 => by rw [prod_range_succ _ n, mul_right_comm, ← prod_range_succ' _ n, prod_range_succ]
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem eventually_constant_prod {u : ℕ → β} {N : ℕ} (hu : ∀ n ≥ N, u n = 1) {n : ℕ} (hn : N ≤ n) :
    (∏ k ∈ range n, u k) = ∏ k ∈ range N, u k := by
  /-
    β : Type u_4
    inst✝ : CommMonoid β
    u : Nat → β
    N : Nat
    hu : ∀ (n : Nat), GE.ge n N → Eq (u n) 1
    n : Nat
    hn : LE.le N n
    ⊢ Eq ((Finset.range n).prod fun k => u k) ((Finset.range N).prod fun k => u k)
  -/
  obtain ⟨m, rfl : n = N + m⟩ := Nat.exists_eq_add_of_le hn
  /-
    case intro
    β : Type u_4
    inst✝ : CommMonoid β
    u : Nat → β
    N : Nat
    hu : ∀ (n : Nat), GE.ge n N → Eq (u n) 1
    m : Nat
    hn : LE.le N (HAdd.hAdd N m)
    ⊢ Eq ((Finset.range (HAdd.hAdd N m)).prod fun k => u k) ((Finset.range N).prod …
  -/
  clear hn
  induction m with
  | zero => simp
  | succ m hm => simp [← add_assoc, prod_range_succ, hm, hu]


@[to_additive]
theorem prod_range_add (f : ℕ → β) (n m : ℕ) :
    (∏ x ∈ range (n + m), f x) = (∏ x ∈ range n, f x) * ∏ x ∈ range m, f (n + x) := by
  induction m with
  | zero => simp
  | succ m hm => rw [Nat.add_succ, prod_range_succ, prod_range_succ, hm, mul_assoc]


@[to_additive]
theorem prod_range_add_div_prod_range {α : Type*} [CommGroup α] (f : ℕ → α) (n m : ℕ) :
    (∏ k ∈ range (n + m), f k) / ∏ k ∈ range n, f k = ∏ k ∈ Finset.range m, f (n + k) :=
  div_eq_of_eq_mul' (prod_range_add f n m)


@[to_additive]
                                                                   /-
                                                                     β : Type u_4
                                                                     inst✝ : CommMonoid β
                                                                     f : Nat → β
                                                                     ⊢ Eq ((Finset.range 0).prod fun k => f k) 1
                                                                   -/
theorem prod_range_zero (f : ℕ → β) : ∏ k ∈ range 0, f k = 1 := by rw [range_zero, prod_empty]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive sum_range_one]
theorem prod_range_one (f : ℕ → β) : ∏ k ∈ range 1, f k = f 0 := by
  /-
    β : Type u_4
    inst✝ : CommMonoid β
    f : Nat → β
    ⊢ Eq ((Finset.range 1).prod fun k => f k) (f 0)
  -/
  rw [range_one, prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_list_map_count [DecidableEq α] (l : List α) {M : Type*} [CommMonoid M] (f : α → M) :
    (l.map f).prod = ∏ m ∈ l.toFinset, f m ^ l.count m := by
  induction l with
  | nil => simp only [map_nil, prod_nil, count_nil, pow_zero, prod_const_one]
  | cons a s IH =>
  simp only [List.map, List.prod_cons, toFinset_cons, IH]
  by_cases has : a ∈ s.toFinset
  · rw [insert_eq_of_mem has, ← insert_erase has, prod_insert (not_mem_erase _ _),
      prod_insert (not_mem_erase _ _), ← mul_assoc, count_cons_self, pow_succ']
    congr 1
    refine prod_congr rfl fun x hx => ?_
    rw [count_cons_of_ne (ne_of_mem_erase hx)]
  rw [prod_insert has, count_cons_self, count_eq_zero_of_not_mem (mt mem_toFinset.2 has), pow_one]
  congr 1
  refine prod_congr rfl fun x hx => ?_
  rw [count_cons_of_ne]
  rintro rfl
  exact has hx


@[to_additive]
theorem prod_list_count [DecidableEq α] [CommMonoid α] (s : List α) :
                                                   /-
                                                     α : Type u_3
                                                     inst✝¹ : DecidableEq α
                                                     inst✝ : CommMonoid α
                                                     s : List α
                                                     ⊢ Eq s.prod (s.toFinset.prod fun m => HPow.hPow m (List.count m s))
                                                   -/
    s.prod = ∏ m ∈ s.toFinset, m ^ s.count m := by simpa using prod_list_map_count s id
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive]
theorem prod_list_count_of_subset [DecidableEq α] [CommMonoid α] (m : List α) (s : Finset α)
    (hs : m.toFinset ⊆ s) : m.prod = ∏ i ∈ s, i ^ m.count i := by
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : List α
    s : Finset α
    hs : HasSubset.Subset m.toFinset s
    ⊢ Eq m.prod (s.prod fun i => HPow.hPow i (List.count i m))
  -/
  rw [prod_list_count]
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : List α
    s : Finset α
    hs : HasSubset.Subset m.toFinset s
    ⊢ Eq (m.toFinset.prod fun m_1 => HPow.hPow m_1 (List.count m_1 m)) (s.prod fun …
  -/
  refine prod_subset hs fun x _ hx => ?_
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : List α
    s : Finset α
    hs : HasSubset.Subset m.toFinset s
    x : α
    x✝ : Membership.mem s x
    hx : Not (Membership.mem m.toFinset x)
    ⊢ Eq (HPow.hPow x (List.count x m)) 1
  -/
  rw [mem_toFinset] at hx
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : List α
    s : Finset α
    hs : HasSubset.Subset m.toFinset s
    x : α
    x✝ : Membership.mem s x
    hx : Not (Membership.mem m x)
    ⊢ Eq (HPow.hPow x (List.count x m)) 1
  -/
  rw [count_eq_zero_of_not_mem hx, pow_zero]
  /-
    🎉 no goals
  -/


theorem sum_filter_count_eq_countP [DecidableEq α] (p : α → Prop) [DecidablePred p] (l : List α) :
    ∑ x ∈ l.toFinset with p x, l.count x = l.countP p := by
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    p : α → Prop
    inst✝ : DecidablePred p
    l : List α
    ⊢ Eq ((Finset.filter (fun x => p x) l.toFinset).sum fun x => List.count x l) ( …
  -/
  simp [Finset.sum, sum_map_count_dedup_filter_eq_countP p l]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_multiset_map_count [DecidableEq α] (s : Multiset α) {M : Type*} [CommMonoid M]
    (f : α → M) : (s.map f).prod = ∏ m ∈ s.toFinset, f m ^ s.count m := by
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    s : Multiset α
    M : Type u_6
    inst✝ : CommMonoid M
    f : α → M
    ⊢ Eq (Multiset.map f s).prod (s.toFinset.prod fun m => HPow.hPow (f m) (Multis …
  -/
  refine Quot.induction_on s fun l => ?_
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    s : Multiset α
    M : Type u_6
    inst✝ : CommMonoid M
    f : α → M
    l : List α
    ⊢ Eq (Multiset.map f (Quot.mk (⇑(List.isSetoid α)) l)).prod ((Multiset.toFinse …
  -/
  simp [prod_list_map_count l f]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_multiset_count [DecidableEq α] [CommMonoid α] (s : Multiset α) :
    s.prod = ∏ m ∈ s.toFinset, m ^ s.count m := by
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    s : Multiset α
    ⊢ Eq s.prod (s.toFinset.prod fun m => HPow.hPow m (Multiset.count m s))
  -/
  convert prod_multiset_map_count s id
  /-
    case h.e'_2.h.e'_3
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    s : Multiset α
    ⊢ Eq s (Multiset.map id s)
  -/
  rw [Multiset.map_id]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_multiset_count_of_subset [DecidableEq α] [CommMonoid α] (m : Multiset α) (s : Finset α)
    (hs : m.toFinset ⊆ s) : m.prod = ∏ i ∈ s, i ^ m.count i := by
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    s : Finset α
    hs : HasSubset.Subset m.toFinset s
    ⊢ Eq m.prod (s.prod fun i => HPow.hPow i (Multiset.count i m))
  -/
  revert hs
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    s : Finset α
    ⊢ HasSubset.Subset m.toFinset s → Eq m.prod (s.prod fun i => HPow.hPow i (Mult …
  -/
  refine Quot.induction_on m fun l => ?_
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    s : Finset α
    l : List α
    ⊢ HasSubset.Subset (Multiset.toFinset (Quot.mk (⇑(List.isSetoid α)) l)) s → Eq …
  -/
  simp only [quot_mk_to_coe'', prod_coe, coe_count]
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    s : Finset α
    l : List α
    ⊢ HasSubset.Subset (↑l).toFinset s → Eq l.prod (s.prod fun x => HPow.hPow x (L …
  -/
  apply prod_list_count_of_subset l s
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_mem_multiset [DecidableEq α] (m : Multiset α) (f : { x // x ∈ m } → β) (g : α → β)
    (hfg : ∀ x, f x = g x) : ∏ x : { x // x ∈ m }, f x = ∏ x ∈ m.toFinset, g x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    m : Multiset α
    f : (Subtype fun x => Membership.mem m x) → β
    g : α → β
    hfg : ∀ (x : Subtype fun x => Membership.mem m x), Eq (f x) (g ↑x)
    ⊢ Eq (Finset.univ.prod fun x => f x) (m.toFinset.prod fun x => g x)
  -/
  refine prod_bij' (fun x _ ↦ x) (fun x hx ↦ ⟨x, Multiset.mem_toFinset.1 hx⟩) ?_ ?_ ?_ ?_ ?_ <;>
    /-
      case refine_1
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      inst✝ : DecidableEq α
      m : Multiset α
      f : (Subtype fun x => Membership.mem m x) → β
      g : α → β
      hfg : ∀ (x : Subtype fun x => Membership.mem m x), Eq (f x) (g ↑x)
      ⊢ ∀ (a : Subtype fun x => Membership.mem m x) (ha : Membership.mem Finset.univ …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [hfg]
    /-
      🎉 no goals
    -/


/-- To prove a property of a product, it suffices to prove that
the property is multiplicative and holds on factors. -/
@[to_additive "To prove a property of a sum, it suffices to prove that
the property is additive and holds on summands."]
theorem prod_induction {M : Type*} [CommMonoid M] (f : α → M) (p : M → Prop)
    (hom : ∀ a b, p a → p b → p (a * b)) (unit : p 1) (base : ∀ x ∈ s, p <| f x) :
    p <| ∏ x ∈ s, f x :=
  Multiset.prod_induction _ _ hom unit (Multiset.forall_mem_map_iff.mpr base)


/-- To prove a property of a product, it suffices to prove that
the property is multiplicative and holds on factors. -/
@[to_additive "To prove a property of a sum, it suffices to prove that
the property is additive and holds on summands."]
theorem prod_induction_nonempty {M : Type*} [CommMonoid M] (f : α → M) (p : M → Prop)
    (hom : ∀ a b, p a → p b → p (a * b)) (nonempty : s.Nonempty) (base : ∀ x ∈ s, p <| f x) :
    p <| ∏ x ∈ s, f x :=
                                             /-
                                               α : Type u_3
                                               s : Finset α
                                               M : Type u_6
                                               inst✝ : CommMonoid M
                                               f : α → M
                                               p : M → Prop
                                               hom : ∀ (a b : M), p a → p b → p (HMul.hMul a b)
                                               nonempty : s.Nonempty
                                               base : ∀ (x : α), Membership.mem s x → p (f x)
                                               ⊢ Ne (Multiset.map (fun x => f x) s.val) EmptyCollection.emptyCollection
                                             -/
  Multiset.prod_induction_nonempty p hom (by simp [nonempty_iff_ne_empty.mp nonempty])
                                             /-
                                               🎉 no goals
                                             -/
    (Multiset.forall_mem_map_iff.mpr base)


/-- For any product along `{0, ..., n - 1}` of a commutative-monoid-valued function, we can verify
that it's equal to a different function just by checking ratios of adjacent terms.

This is a multiplicative discrete analogue of the fundamental theorem of calculus. -/
@[to_additive "For any sum along `{0, ..., n - 1}` of a commutative-monoid-valued function, we can
verify that it's equal to a different function just by checking differences of adjacent terms.

This is a discrete analogue of the fundamental theorem of calculus."]
theorem prod_range_induction (f s : ℕ → β) (base : s 0 = 1)
    (step : ∀ n, s (n + 1) = s n * f n) (n : ℕ) :
    ∏ k ∈ Finset.range n, f k = s n := by
  induction n with
  | zero => rw [Finset.prod_range_zero, base]
  | succ k hk => simp only [hk, Finset.prod_range_succ, step, mul_comm]


/-- A telescoping product along `{0, ..., n - 1}` of a commutative group valued function reduces to
the ratio of the last and first factors. -/
@[to_additive "A telescoping sum along `{0, ..., n - 1}` of an additive commutative group valued
function reduces to the difference of the last and first terms."]
theorem prod_range_div {M : Type*} [CommGroup M] (f : ℕ → M) (n : ℕ) :
                                                       /-
                                                         M : Type u_6
                                                         inst✝ : CommGroup M
                                                         f : Nat → M
                                                         n : Nat
                                                         ⊢ Eq ((Finset.range n).prod fun i => HDiv.hDiv (f (HAdd.hAdd i 1)) (f i)) (HDi …
                                                       -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    (∏ i ∈ range n, f (i + 1) / f i) = f n / f 0 := by apply prod_range_induction <;> simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[to_additive]
theorem prod_range_div' {M : Type*} [CommGroup M] (f : ℕ → M) (n : ℕ) :
                                                       /-
                                                         M : Type u_6
                                                         inst✝ : CommGroup M
                                                         f : Nat → M
                                                         n : Nat
                                                         ⊢ Eq ((Finset.range n).prod fun i => HDiv.hDiv (f i) (f (HAdd.hAdd i 1))) (HDi …
                                                       -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    (∏ i ∈ range n, f i / f (i + 1)) = f 0 / f n := by apply prod_range_induction <;> simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[to_additive]
theorem eq_prod_range_div {M : Type*} [CommGroup M] (f : ℕ → M) (n : ℕ) :
                                                     /-
                                                       M : Type u_6
                                                       inst✝ : CommGroup M
                                                       f : Nat → M
                                                       n : Nat
                                                       ⊢ Eq (f n) (HMul.hMul (f 0) ((Finset.range n).prod fun i => HDiv.hDiv (f (HAdd …
                                                     -/
    f n = f 0 * ∏ i ∈ range n, f (i + 1) / f i := by rw [prod_range_div, mul_div_cancel]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
theorem eq_prod_range_div' {M : Type*} [CommGroup M] (f : ℕ → M) (n : ℕ) :
    f n = ∏ i ∈ range (n + 1), if i = 0 then f 0 else f i / f (i - 1) := by
  /-
    M : Type u_6
    inst✝ : CommGroup M
    f : Nat → M
    n : Nat
    ⊢ Eq (f n) ((Finset.range (HAdd.hAdd n 1)).prod fun i => ite (Eq i 0) (f 0) (H …
  -/
  conv_lhs => rw [Finset.eq_prod_range_div f]
  /-
    M : Type u_6
    inst✝ : CommGroup M
    f : Nat → M
    n : Nat
    ⊢ Eq (HMul.hMul (f 0) ((Finset.range n).prod fun i => HDiv.hDiv (f (HAdd.hAdd  …
  -/
  simp [Finset.prod_range_succ', mul_comm]
  /-
    🎉 no goals
  -/


/-- A telescoping sum along `{0, ..., n-1}` of an `ℕ`-valued function
reduces to the difference of the last and first terms
when the function we are summing is monotone.
-/
theorem sum_range_tsub [AddCommMonoid α] [PartialOrder α] [Sub α] [OrderedSub α]
    [AddLeftMono α] [AddLeftReflectLE α] [ExistsAddOfLE α]
    {f : ℕ → α} (h : Monotone f) (n : ℕ) :
    ∑ i ∈ range n, (f (i + 1) - f i) = f n - f 0 := by
  /-
    α : Type u_3
    inst✝⁶ : AddCommMonoid α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Sub α
    inst✝³ : OrderedSub α
    inst✝² : AddLeftMono α
    inst✝¹ : AddLeftReflectLE α
    inst✝ : ExistsAddOfLE α
    f : Nat → α
    h : Monotone f
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HSub.hSub (f (HAdd.hAdd i 1)) (f i)) (HSub …
  -/
  apply sum_range_induction
  /-
    case base
    α : Type u_3
    inst✝⁶ : AddCommMonoid α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Sub α
    inst✝³ : OrderedSub α
    inst✝² : AddLeftMono α
    inst✝¹ : AddLeftReflectLE α
    inst✝ : ExistsAddOfLE α
    f : Nat → α
    h : Monotone f
    n : Nat
    ⊢ Eq (HSub.hSub (f 0) (f 0)) 0
  -/
  case base => apply tsub_eq_of_eq_add; rw [zero_add]
  case step =>
    intro n
    have h₁ : f n ≤ f (n + 1) := h (Nat.le_succ _)
    have h₂ : f 0 ≤ f n := h (Nat.zero_le _)
    rw [tsub_add_eq_add_tsub h₂, add_tsub_cancel_of_le h₁]


theorem sum_tsub_distrib [AddCommMonoid α] [PartialOrder α] [ExistsAddOfLE α]
    [CovariantClass α α (· + ·) (· ≤ ·)] [ContravariantClass α α (· + ·) (· ≤ ·)] [Sub α]
    [OrderedSub α] (s : Finset ι) {f g : ι → α} (hfg : ∀ x ∈ s, g x ≤ f x) :
    ∑ x ∈ s, (f x - g x) = ∑ x ∈ s, f x - ∑ x ∈ s, g x := sum_map_tsub _ hfg


@[to_additive (attr := simp)]
theorem prod_const (b : β) : ∏ _x ∈ s, b = b ^ #s :=
  (congr_arg _ <| s.val.map_const b).trans <| Multiset.prod_replicate #s b


@[to_additive sum_eq_card_nsmul]
theorem prod_eq_pow_card {b : β} (hf : ∀ a ∈ s, f a = b) : ∏ a ∈ s, f a = b ^ #s :=
  (prod_congr rfl hf).trans <| prod_const _


@[to_additive card_nsmul_add_sum]
theorem pow_card_mul_prod {b : β} : b ^ #s * ∏ a ∈ s, f a = ∏ a ∈ s, b * f a :=
  (Finset.prod_const b).symm ▸ prod_mul_distrib.symm


@[to_additive sum_add_card_nsmul]
theorem prod_mul_pow_card {b : β} : (∏ a ∈ s, f a) * b ^ #s = ∏ a ∈ s, f a * b :=
  (Finset.prod_const b).symm ▸ prod_mul_distrib.symm


@[to_additive]
                                                                         /-
                                                                           β : Type u_4
                                                                           inst✝ : CommMonoid β
                                                                           b : β
                                                                           ⊢ ∀ (n : Nat), Eq (HPow.hPow b n) ((Finset.range n).prod fun _k => b)
                                                                         -/
theorem pow_eq_prod_const (b : β) : ∀ n, b ^ n = ∏ _k ∈ range n, b := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[to_additive]
theorem prod_pow (s : Finset α) (n : ℕ) (f : α → β) : ∏ x ∈ s, f x ^ n = (∏ x ∈ s, f x) ^ n :=
  Multiset.prod_map_pow


@[to_additive sum_nsmul_assoc]
lemma prod_pow_eq_pow_sum (s : Finset ι) (f : ι → ℕ) (a : β) :
    ∏ i ∈ s, a ^ f i = a ^ ∑ i ∈ s, f i :=
                     /-
                       ι : Type u_1
                       β : Type u_4
                       inst✝ : CommMonoid β
                       s : Finset ι
                       f : ι → Nat
                       a : β
                       ⊢ Eq (EmptyCollection.emptyCollection.prod fun i => HPow.hPow a (f i)) (HPow.h …
                     -/
                     /-
                       🎉 no goals
                     -/
  cons_induction (by simp) (fun _ _ _ _ ↦ by simp [prod_cons, sum_cons, pow_add, *]) s
                                             /-
                                               🎉 no goals
                                             -/


/-- A product over `Finset.powersetCard` which only depends on the size of the sets is constant. -/
@[to_additive
"A sum over `Finset.powersetCard` which only depends on the size of the sets is constant."]
lemma prod_powersetCard (n : ℕ) (s : Finset α) (f : ℕ → β) :
    ∏ t ∈ powersetCard n s, f #t = f n ^ (#s).choose n := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    n : Nat
    s : Finset α
    f : Nat → β
    ⊢ Eq ((Finset.powersetCard n s).prod fun t => f t.card) (HPow.hPow (f n) (s.ca …
  -/
  rw [prod_eq_pow_card, card_powersetCard]; rintro a ha; rw [(mem_powersetCard.1 ha).2]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
theorem prod_flip {n : ℕ} (f : ℕ → β) :
    (∏ r ∈ range (n + 1), f (n - r)) = ∏ k ∈ range (n + 1), f k := by
  induction n with
  | zero => rw [prod_range_one, prod_range_one]
  | succ n ih =>
    rw [prod_range_succ', prod_range_succ _ (Nat.succ n)]
    simp [← ih]


/-- The difference with `Finset.prod_ninvolution` is that the involution is allowed to use
membership of the domain of the product, rather than being a non-dependent function. -/
@[to_additive "The difference with `Finset.sum_ninvolution` is that the involution is allowed to use
membership of the domain of the sum, rather than being a non-dependent function."]
lemma prod_involution (g : ∀ a ∈ s, α) (hg₁ : ∀ a ha, f a * f (g a ha) = 1)
    (hg₃ : ∀ a ha, f a ≠ 1 → g a ha ≠ a)
    (g_mem : ∀ a ha, g a ha ∈ s) (hg₄ : ∀ a ha, g (g a ha) (g_mem a ha) = a) :
    ∏ x ∈ s, f x = 1 := by
  classical
  induction s using Finset.strongInduction with | H s ih => ?_
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
  · simp
  have : {x, g x hx} ⊆ s := by simp [insert_subset_iff, hx, g_mem]
  suffices h : ∏ x ∈ s \ {x, g x hx}, f x = 1 by
    rw [← prod_sdiff this, h, one_mul]
    cases eq_or_ne (g x hx) x with
    | inl hx' => simpa [hx'] using hg₃ x hx
    | inr hx' => rw [prod_pair hx'.symm, hg₁]
  suffices h₃ : ∀ a (ha : a ∈ s \ {x, g x hx}), g a (sdiff_subset ha) ∈ s \ {x, g x hx} from
    ih (s \ {x, g x hx}) (ssubset_iff.2 ⟨x, by simp [insert_subset_iff, hx]⟩) _
      (by simp [hg₁]) (fun _ _ => hg₃ _ _) h₃ (fun _ _ => hg₄ _ _)
  simp only [mem_sdiff, mem_insert, mem_singleton, not_or, g_mem, true_and]
  rintro a ⟨ha₁, ha₂, ha₃⟩
  refine ⟨fun h => by simp [← h, hg₄] at ha₃, fun h => ?_⟩
  have : g (g a ha₁) (g_mem _ _) = g (g x hx) (g_mem _ _) := by simp only [h]
  exact ha₂ (by simpa [hg₄] using this)


/-- The difference with `Finset.prod_involution` is that the involution is a non-dependent function,
rather than being allowed to use membership of the domain of the product. -/
@[to_additive "The difference with `Finset.sum_involution` is that the involution is a non-dependent
function, rather than being allowed to use membership of the domain of the sum."]
lemma prod_ninvolution (g : α → α) (hg₁ : ∀ a, f a * f (g a) = 1) (hg₂ : ∀ a, f a ≠ 1 → g a ≠ a)
    (g_mem : ∀ a, g a ∈ s) (hg₃ : ∀ a, g (g a) = a) : ∏ x ∈ s, f x = 1 :=
  prod_involution (fun i _ => g i) (fun i _ => hg₁ i) (fun _ _ hi => hg₂ _ hi)
    (fun i _ => g_mem i) (fun i _ => hg₃ i)


/-- The product of the composition of functions `f` and `g`, is the product over `b ∈ s.image g` of
`f b` to the power of the cardinality of the fibre of `b`. See also `Finset.prod_image`. -/
@[to_additive "The sum of the composition of functions `f` and `g`, is the sum over `b ∈ s.image g`
of `f b` times of the cardinality of the fibre of `b`. See also `Finset.sum_image`."]
theorem prod_comp [DecidableEq γ] (f : γ → β) (g : α → γ) :
    ∏ a ∈ s, f (g a) = ∏ b ∈ s.image g, f b ^ #{a ∈ s | g a = b} := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    s : Finset α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq γ
    f : γ → β
    g : α → γ
    ⊢ Eq (s.prod fun a => f (g a)) ((Finset.image g s).prod fun b => HPow.hPow (f  …
  -/
  simp_rw [← prod_const, prod_fiberwise_of_maps_to' fun _ ↦ mem_image_of_mem _]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_piecewise [DecidableEq α] (s t : Finset α) (f g : α → β) :
    (∏ x ∈ s, (t.piecewise f g) x) = (∏ x ∈ s ∩ t, f x) * ∏ x ∈ s \ t, g x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s t : Finset α
    f g : α → β
    ⊢ Eq (s.prod fun x => t.piecewise f g x) (HMul.hMul ((Inter.inter s t).prod fu …
  -/
  erw [prod_ite, filter_mem_eq_inter, ← sdiff_eq_filter]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_inter_mul_prod_diff [DecidableEq α] (s t : Finset α) (f : α → β) :
    (∏ x ∈ s ∩ t, f x) * ∏ x ∈ s \ t, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s t : Finset α
    f : α → β
    ⊢ Eq (HMul.hMul ((Inter.inter s t).prod fun x => f x) ((SDiff.sdiff s t).prod  …
  -/
  convert (s.prod_piecewise t f f).symm
  /-
    case h.e'_3.a.h.e
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s t : Finset α
    f : α → β
    x✝ : α
    a✝ : Membership.mem s x✝
    ⊢ Eq f (t.piecewise f f)
  -/
  simp (config := { unfoldPartialApp := true }) [Finset.piecewise]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_eq_mul_prod_diff_singleton [DecidableEq α] {s : Finset α} {i : α} (h : i ∈ s)
    (f : α → β) : ∏ x ∈ s, f x = f i * ∏ x ∈ s \ {i}, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Membership.mem s i
    f : α → β
    ⊢ Eq (s.prod fun x => f x) (HMul.hMul (f i) ((SDiff.sdiff s (Singleton.singlet …
  -/
  convert (s.prod_inter_mul_prod_diff {i} f).symm
  /-
    case h.e'_3.h.e'_5
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Membership.mem s i
    f : α → β
    ⊢ Eq (f i) ((Inter.inter s (Singleton.singleton i)).prod fun x => f x)
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_eq_prod_diff_singleton_mul [DecidableEq α] {s : Finset α} {i : α} (h : i ∈ s)
    (f : α → β) : ∏ x ∈ s, f x = (∏ x ∈ s \ {i}, f x) * f i := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Membership.mem s i
    f : α → β
    ⊢ Eq (s.prod fun x => f x) (HMul.hMul ((SDiff.sdiff s (Singleton.singleton i)) …
  -/
  rw [prod_eq_mul_prod_diff_singleton h, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem _root_.Fintype.prod_eq_mul_prod_compl [DecidableEq α] [Fintype α] (a : α) (f : α → β) :
    ∏ i, f i = f a * ∏ i ∈ {a}ᶜ, f i :=
  prod_eq_mul_prod_diff_singleton (mem_univ a) f


@[to_additive]
theorem _root_.Fintype.prod_eq_prod_compl_mul [DecidableEq α] [Fintype α] (a : α) (f : α → β) :
    ∏ i, f i = (∏ i ∈ {a}ᶜ, f i) * f a :=
  prod_eq_prod_diff_singleton_mul (mem_univ a) f


theorem dvd_prod_of_mem (f : α → β) {a : α} {s : Finset α} (ha : a ∈ s) : f a ∣ ∏ i ∈ s, f i := by
  classical
    rw [Finset.prod_eq_mul_prod_diff_singleton ha]
    exact dvd_mul_right _ _


/-- A product can be partitioned into a product of products, each equivalent under a setoid. -/
@[to_additive "A sum can be partitioned into a sum of sums, each equivalent under a setoid."]
theorem prod_partition (R : Setoid α) [DecidableRel R.r] :
    ∏ x ∈ s, f x = ∏ xbar ∈ s.image (Quotient.mk _), ∏ y ∈ s with ⟦y⟧ = xbar, f y := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    ⊢ Eq (s.prod fun x => f x) ((Finset.image (Quotient.mk R) s).prod fun xbar =>  …
  -/
  refine (Finset.prod_image' f fun x _hx => ?_).symm
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    x : α
    _hx : Membership.mem s x
    ⊢ Eq ((Finset.filter (fun y => Eq (Quotient.mk R y) (Quotient.mk R x)) s).prod …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If we can partition a product into subsets that cancel out, then the whole product cancels. -/
@[to_additive "If we can partition a sum into subsets that cancel out, then the whole sum cancels."]
theorem prod_cancels_of_partition_cancels (R : Setoid α) [DecidableRel R]
    (h : ∀ x ∈ s, ∏ a ∈ s with R a x, f a = 1) : ∏ x ∈ s, f x = 1 := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    h : ∀ (x : α), Membership.mem s x → Eq ((Finset.filter (fun a => R a x) s).pro …
    ⊢ Eq (s.prod fun x => f x) 1
  -/
  rw [prod_partition R, ← Finset.prod_eq_one]
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    h : ∀ (x : α), Membership.mem s x → Eq ((Finset.filter (fun a => R a x) s).pro …
    ⊢ ∀ (x : Quotient R), Membership.mem (Finset.image (Quotient.mk R) s) x → Eq ( …
  -/
  intro xbar xbar_in_s
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    h : ∀ (x : α), Membership.mem s x → Eq ((Finset.filter (fun a => R a x) s).pro …
    xbar : Quotient R
    xbar_in_s : Membership.mem (Finset.image (Quotient.mk R) s) xbar
    ⊢ Eq ((Finset.filter (fun y => Eq (Quotient.mk R y) xbar) s).prod fun y => f y …
  -/
  obtain ⟨x, x_in_s, rfl⟩ := mem_image.mp xbar_in_s
  /-
    case intro.intro
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    h : ∀ (x : α), Membership.mem s x → Eq ((Finset.filter (fun a => R a x) s).pro …
    x : α
    x_in_s : Membership.mem s x
    xbar_in_s : Membership.mem (Finset.image (Quotient.mk R) s) (Quotient.mk R x)
    ⊢ Eq ((Finset.filter (fun y => Eq (Quotient.mk R y) (Quotient.mk R x)) s).prod …
  -/
  simp only [← Quotient.eq] at h
  /-
    case intro.intro
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommMonoid β
    R : Setoid α
    inst✝ : DecidableRel ⇑R
    x : α
    x_in_s : Membership.mem s x
    xbar_in_s : Membership.mem (Finset.image (Quotient.mk R) s) (Quotient.mk R x)
    h : ∀ (x : α), Membership.mem s x → Eq ((Finset.filter (fun a => Eq (Quotient. …
    ⊢ Eq ((Finset.filter (fun y => Eq (Quotient.mk R y) (Quotient.mk R x)) s).prod …
  -/
  exact h x x_in_s
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_update_of_not_mem [DecidableEq α] {s : Finset α} {i : α} (h : i ∉ s) (f : α → β)
    (b : β) : ∏ x ∈ s, Function.update f i b x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Not (Membership.mem s i)
    f : α → β
    b : β
    ⊢ Eq (s.prod fun x => Function.update f i b x) (s.prod fun x => f x)
  -/
  apply prod_congr rfl
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Not (Membership.mem s i)
    f : α → β
    b : β
    ⊢ ∀ (x : α), Membership.mem s x → Eq (Function.update f i b x) (f x)
  -/
  intros j hj
  have : j ≠ i := by
    rintro rfl
    exact h hj
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Not (Membership.mem s i)
    f : α → β
    b : β
    j : α
    hj : Membership.mem s j
    this : Ne j i
    ⊢ Eq (Function.update f i b j) (f j)
  -/
  simp [this]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_update_of_mem [DecidableEq α] {s : Finset α} {i : α} (h : i ∈ s) (f : α → β) (b : β) :
    ∏ x ∈ s, Function.update f i b x = b * ∏ x ∈ s \ singleton i, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Membership.mem s i
    f : α → β
    b : β
    ⊢ Eq (s.prod fun x => Function.update f i b x) (HMul.hMul b ((SDiff.sdiff s (S …
  -/
  rw [update_eq_piecewise, prod_piecewise]
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    i : α
    h : Membership.mem s i
    f : α → β
    b : β
    ⊢ Eq (HMul.hMul ((Inter.inter s (Singleton.singleton i)).prod fun x => b) ((SD …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


/-- If a product of a `Finset` of size at most 1 has a given value, so
do the terms in that product. -/
@[to_additive eq_of_card_le_one_of_sum_eq "If a sum of a `Finset` of size at most 1 has a given
value, so do the terms in that sum."]
theorem eq_of_card_le_one_of_prod_eq {s : Finset α} (hc : #s ≤ 1) {f : α → β} {b : β}
    (h : ∏ x ∈ s, f x = b) : ∀ x ∈ s, f x = b := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    hc : LE.le s.card 1
    f : α → β
    b : β
    h : Eq (s.prod fun x => f x) b
    ⊢ ∀ (x : α), Membership.mem s x → Eq (f x) b
  -/
  intro x hx
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    hc : LE.le s.card 1
    f : α → β
    b : β
    h : Eq (s.prod fun x => f x) b
    x : α
    hx : Membership.mem s x
    ⊢ Eq (f x) b
  -/
  by_cases hc0 : #s = 0
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      h : Eq (s.prod fun x => f x) b
      x : α
      hx : Membership.mem s x
      hc0 : Eq s.card 0
      ⊢ Eq (f x) b
    -/
  · exact False.elim (card_ne_zero_of_mem hx hc0)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      h : Eq (s.prod fun x => f x) b
      x : α
      hx : Membership.mem s x
      hc0 : Not (Eq s.card 0)
      ⊢ Eq (f x) b
    -/
  · have h1 : #s = 1 := le_antisymm hc (Nat.one_le_of_lt (Nat.pos_of_ne_zero hc0))
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      h : Eq (s.prod fun x => f x) b
      x : α
      hx : Membership.mem s x
      hc0 : Not (Eq s.card 0)
      h1 : Eq s.card 1
      ⊢ Eq (f x) b
    -/
    rw [card_eq_one] at h1
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      h : Eq (s.prod fun x => f x) b
      x : α
      hx : Membership.mem s x
      hc0 : Not (Eq s.card 0)
      h1 : Exists fun a => Eq s (Singleton.singleton a)
      ⊢ Eq (f x) b
    -/
    obtain ⟨x2, hx2⟩ := h1
    /-
      case neg.intro
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      h : Eq (s.prod fun x => f x) b
      x : α
      hx : Membership.mem s x
      hc0 : Not (Eq s.card 0)
      x2 : α
      hx2 : Eq s (Singleton.singleton x2)
      ⊢ Eq (f x) b
    -/
    rw [hx2, mem_singleton] at hx
    /-
      case neg.intro
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      h : Eq (s.prod fun x => f x) b
      x : α
      hc0 : Not (Eq s.card 0)
      x2 : α
      hx : Eq x x2
      hx2 : Eq s (Singleton.singleton x2)
      ⊢ Eq (f x) b
    -/
    simp_rw [hx2] at h
    /-
      case neg.intro
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      x : α
      hc0 : Not (Eq s.card 0)
      x2 : α
      hx : Eq x x2
      hx2 : Eq s (Singleton.singleton x2)
      h : Eq ((Singleton.singleton x2).prod fun x => f x) b
      ⊢ Eq (f x) b
    -/
    rw [hx]
    /-
      case neg.intro
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      x : α
      hc0 : Not (Eq s.card 0)
      x2 : α
      hx : Eq x x2
      hx2 : Eq s (Singleton.singleton x2)
      h : Eq ((Singleton.singleton x2).prod fun x => f x) b
      ⊢ Eq (f x2) b
    -/
    rw [prod_singleton] at h
    /-
      case neg.intro
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      s : Finset α
      hc : LE.le s.card 1
      f : α → β
      b : β
      x : α
      hc0 : Not (Eq s.card 0)
      x2 : α
      hx : Eq x x2
      hx2 : Eq s (Singleton.singleton x2)
      h : Eq (f x2) b
      ⊢ Eq (f x2) b
    -/
    exact h
    /-
      🎉 no goals
    -/


/-- Taking a product over `s : Finset α` is the same as multiplying the value on a single element
`f a` by the product of `s.erase a`.

See `Multiset.prod_map_erase` for the `Multiset` version. -/
@[to_additive "Taking a sum over `s : Finset α` is the same as adding the value on a single element
`f a` to the sum over `s.erase a`.

See `Multiset.sum_map_erase` for the `Multiset` version."]
theorem mul_prod_erase [DecidableEq α] (s : Finset α) (f : α → β) {a : α} (h : a ∈ s) :
    (f a * ∏ x ∈ s.erase a, f x) = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    f : α → β
    a : α
    h : Membership.mem s a
    ⊢ Eq (HMul.hMul (f a) ((s.erase a).prod fun x => f x)) (s.prod fun x => f x)
  -/
  rw [← prod_insert (not_mem_erase a s), insert_erase h]
  /-
    🎉 no goals
  -/


/-- A variant of `Finset.mul_prod_erase` with the multiplication swapped. -/
@[to_additive "A variant of `Finset.add_sum_erase` with the addition swapped."]
theorem prod_erase_mul [DecidableEq α] (s : Finset α) (f : α → β) {a : α} (h : a ∈ s) :
                                                      /-
                                                        α : Type u_3
                                                        β : Type u_4
                                                        inst✝¹ : CommMonoid β
                                                        inst✝ : DecidableEq α
                                                        s : Finset α
                                                        f : α → β
                                                        a : α
                                                        h : Membership.mem s a
                                                        ⊢ Eq (HMul.hMul ((s.erase a).prod fun x => f x) (f a)) (s.prod fun x => f x)
                                                      -/
    (∏ x ∈ s.erase a, f x) * f a = ∏ x ∈ s, f x := by rw [mul_comm, mul_prod_erase s f h]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If a function applied at a point is 1, a product is unchanged by
removing that point, if present, from a `Finset`. -/
@[to_additive "If a function applied at a point is 0, a sum is unchanged by
removing that point, if present, from a `Finset`."]
theorem prod_erase [DecidableEq α] (s : Finset α) {f : α → β} {a : α} (h : f a = 1) :
    ∏ x ∈ s.erase a, f x = ∏ x ∈ s, f x := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    f : α → β
    a : α
    h : Eq (f a) 1
    ⊢ Eq ((s.erase a).prod fun x => f x) (s.prod fun x => f x)
  -/
  rw [← sdiff_singleton_eq_erase]
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    f : α → β
    a : α
    h : Eq (f a) 1
    ⊢ Eq ((SDiff.sdiff s (Singleton.singleton a)).prod fun x => f x) (s.prod fun x …
  -/
  refine prod_subset sdiff_subset fun x hx hnx => ?_
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    f : α → β
    a : α
    h : Eq (f a) 1
    x : α
    hx : Membership.mem s x
    hnx : Not (Membership.mem (SDiff.sdiff s (Singleton.singleton a)) x)
    ⊢ Eq (f x) 1
  -/
  rw [sdiff_singleton_eq_erase] at hnx
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq α
    s : Finset α
    f : α → β
    a : α
    h : Eq (f a) 1
    x : α
    hx : Membership.mem s x
    hnx : Not (Membership.mem (s.erase a) x)
    ⊢ Eq (f x) 1
  -/
  rwa [eq_of_mem_of_not_mem_erase hx hnx]
  /-
    🎉 no goals
  -/


/-- See also `Finset.prod_ite_zero`. -/
@[to_additive "See also `Finset.sum_boole`."]
theorem prod_ite_one (s : Finset α) (p : α → Prop) [DecidablePred p]
    (h : ∀ i ∈ s, ∀ j ∈ s, p i → p j → i = j) (a : β) :
    ∏ i ∈ s, ite (p i) a 1 = ite (∃ i ∈ s, p i) a 1 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : CommMonoid β
    s : Finset α
    p : α → Prop
    inst✝ : DecidablePred p
    h : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j  …
    a : β
    ⊢ Eq (s.prod fun i => ite (p i) a 1) (ite (Exists fun i => And (Membership.mem …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      s : Finset α
      p : α → Prop
      inst✝ : DecidablePred p
      h✝ : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j …
      a : β
      h : Exists fun i => And (Membership.mem s i) (p i)
      ⊢ Eq (s.prod fun i => ite (p i) a 1) a
    -/
  · obtain ⟨i, hi, hpi⟩ := h
    /-
      case pos.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      s : Finset α
      p : α → Prop
      inst✝ : DecidablePred p
      h : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j  …
      a : β
      i : α
      hi : Membership.mem s i
      hpi : p i
      ⊢ Eq (s.prod fun i => ite (p i) a 1) a
    -/
    rw [prod_eq_single_of_mem _ hi, if_pos hpi]
    /-
      case pos.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      s : Finset α
      p : α → Prop
      inst✝ : DecidablePred p
      h : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j  …
      a : β
      i : α
      hi : Membership.mem s i
      hpi : p i
      ⊢ ∀ (b : α), Membership.mem s b → Ne b i → Eq (ite (p b) a 1) 1
    -/
    exact fun j hj hji ↦ if_neg fun hpj ↦ hji <| h _ hj _ hi hpj hpi
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      s : Finset α
      p : α → Prop
      inst✝ : DecidablePred p
      h✝ : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j …
      a : β
      h : Not (Exists fun i => And (Membership.mem s i) (p i))
      ⊢ Eq (s.prod fun i => ite (p i) a 1) 1
    -/
  · push_neg at h
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      s : Finset α
      p : α → Prop
      inst✝ : DecidablePred p
      h✝ : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j …
      a : β
      h : ∀ (i : α), Membership.mem s i → Not (p i)
      ⊢ Eq (s.prod fun i => ite (p i) a 1) 1
    -/
    rw [prod_eq_one]
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid β
      s : Finset α
      p : α → Prop
      inst✝ : DecidablePred p
      h✝ : ∀ (i : α), Membership.mem s i → ∀ (j : α), Membership.mem s j → p i → p j …
      a : β
      h : ∀ (i : α), Membership.mem s i → Not (p i)
      ⊢ ∀ (x : α), Membership.mem s x → Eq (ite (p x) a 1) 1
    -/
    exact fun i hi => if_neg (h i hi)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_erase_lt_of_one_lt {γ : Type*} [DecidableEq α] [CommMonoid γ] [Preorder γ]
    [MulLeftStrictMono γ] {s : Finset α} {d : α} (hd : d ∈ s) {f : α → γ}
    (hdf : 1 < f d) : ∏ m ∈ s.erase d, f m < ∏ m ∈ s, f m := by
  /-
    α : Type u_3
    γ : Type u_6
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid γ
    inst✝¹ : Preorder γ
    inst✝ : MulLeftStrictMono γ
    s : Finset α
    d : α
    hd : Membership.mem s d
    f : α → γ
    hdf : LT.lt 1 (f d)
    ⊢ LT.lt ((s.erase d).prod fun m => f m) (s.prod fun m => f m)
  -/
  conv in ∏ m ∈ s, f m => rw [← Finset.insert_erase hd]
  /-
    α : Type u_3
    γ : Type u_6
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid γ
    inst✝¹ : Preorder γ
    inst✝ : MulLeftStrictMono γ
    s : Finset α
    d : α
    hd : Membership.mem s d
    f : α → γ
    hdf : LT.lt 1 (f d)
    ⊢ LT.lt ((s.erase d).prod fun m => f m) ((Insert.insert d (s.erase d)).prod fu …
  -/
  rw [Finset.prod_insert (Finset.not_mem_erase d s)]
  /-
    α : Type u_3
    γ : Type u_6
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid γ
    inst✝¹ : Preorder γ
    inst✝ : MulLeftStrictMono γ
    s : Finset α
    d : α
    hd : Membership.mem s d
    f : α → γ
    hdf : LT.lt 1 (f d)
    ⊢ LT.lt ((s.erase d).prod fun m => f m) (HMul.hMul (f d) ((s.erase d).prod fun …
  -/
  exact lt_mul_of_one_lt_left' _ hdf
  /-
    🎉 no goals
  -/


/-- If a product is 1 and the function is 1 except possibly at one
point, it is 1 everywhere on the `Finset`. -/
@[to_additive "If a sum is 0 and the function is 0 except possibly at one
point, it is 0 everywhere on the `Finset`."]
theorem eq_one_of_prod_eq_one {s : Finset α} {f : α → β} {a : α} (hp : ∏ x ∈ s, f x = 1)
    (h1 : ∀ x ∈ s, x ≠ a → f x = 1) : ∀ x ∈ s, f x = 1 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    s : Finset α
    f : α → β
    a : α
    hp : Eq (s.prod fun x => f x) 1
    h1 : ∀ (x : α), Membership.mem s x → Ne x a → Eq (f x) 1
    ⊢ ∀ (x : α), Membership.mem s x → Eq (f x) 1
  -/
  intro x hx
  classical
    by_cases h : x = a
    · rw [h]
      rw [h] at hx
      rw [← prod_subset (singleton_subset_iff.2 hx) fun t ht ha => h1 t ht (not_mem_singleton.1 ha),
        prod_singleton] at hp
      exact hp
    · exact h1 x hx h


@[to_additive sum_boole_nsmul]
theorem prod_pow_boole [DecidableEq α] (s : Finset α) (f : α → β) (a : α) :
                                                                 /-
                                                                   α : Type u_3
                                                                   β : Type u_4
                                                                   inst✝¹ : CommMonoid β
                                                                   inst✝ : DecidableEq α
                                                                   s : Finset α
                                                                   f : α → β
                                                                   a : α
                                                                   ⊢ Eq (s.prod fun x => HPow.hPow (f x) (ite (Eq a x) 1 0)) (ite (Membership.mem …
                                                                 -/
    (∏ x ∈ s, f x ^ ite (a = x) 1 0) = ite (a ∈ s) (f a) 1 := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem prod_dvd_prod_of_dvd {S : Finset α} (g1 g2 : α → β) (h : ∀ a ∈ S, g1 a ∣ g2 a) :
    S.prod g1 ∣ S.prod g2 := by
  induction S using Finset.cons_induction with
  | empty => simp
  | cons a T haT IH =>
    rw [Finset.prod_cons, Finset.prod_cons]
    rw [Finset.forall_mem_cons] at h
    exact mul_dvd_mul h.1 <| IH h.2


theorem prod_dvd_prod_of_subset {ι M : Type*} [CommMonoid M] (s t : Finset ι) (f : ι → M)
    (h : s ⊆ t) : (∏ i ∈ s, f i) ∣ ∏ i ∈ t, f i :=
                                                            /-
                                                              ι : Type u_6
                                                              M : Type u_7
                                                              inst✝ : CommMonoid M
                                                              s t : Finset ι
                                                              f : ι → M
                                                              h : HasSubset.Subset s t
                                                              ⊢ LE.le s.val t.val
                                                            -/
  Multiset.prod_dvd_prod_of_le <| Multiset.map_le_map <| by simpa
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive]
lemma prod_mul_eq_prod_mul_of_exists {s : Finset α} {f : α → β} {b₁ b₂ : β}
    (a : α) (ha : a ∈ s) (h : f a * b₁ = f a * b₂) :
    (∏ a ∈ s, f a) * b₁ = (∏ a ∈ s, f a) * b₂ := by
  classical
  rw [← insert_erase ha]
  simp only [mem_erase, ne_eq, not_true_eq_false, false_and, not_false_eq_true, prod_insert]
  rw [mul_assoc, mul_comm, mul_assoc, mul_comm b₁, h, ← mul_assoc, mul_comm _ (f a)]


@[to_additive]
lemma isSquare_prod {s : Finset ι} [CommMonoid α] (f : ι → α)
    (h : ∀ c ∈ s, IsSquare (f c)) : IsSquare (∏ i ∈ s, f i) := by
  /-
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    ⊢ IsSquare (s.prod fun i => f i)
  -/
  rw [isSquare_iff_exists_sq]
  /-
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    ⊢ Exists fun c => Eq (s.prod fun i => f i) (HPow.hPow c 2)
  -/
  use (∏ (x : s), ((isSquare_iff_exists_sq _).mp (h _ x.2)).choose)
  /-
    case h
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    ⊢ Eq (s.prod fun i => f i) (HPow.hPow (Finset.univ.prod fun x => ⋯.choose) 2)
  -/
  rw [@sq, ← Finset.prod_mul_distrib, ← Finset.prod_coe_sort]
  /-
    case h
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    ⊢ Eq (Finset.univ.prod fun i => f ↑i) (Finset.univ.prod fun x => HMul.hMul ⋯.c …
  -/
  congr
  /-
    case h.e_f
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    ⊢ Eq (fun i => f ↑i) fun x => HMul.hMul ⋯.choose ⋯.choose
  -/
  ext i
  /-
    case h.e_f.h
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    i : Subtype fun x => Membership.mem s x
    ⊢ Eq (f ↑i) (HMul.hMul ⋯.choose ⋯.choose)
  -/
  rw [← @sq]
  /-
    case h.e_f.h
    ι : Type u_1
    α : Type u_3
    s : Finset ι
    inst✝ : CommMonoid α
    f : ι → α
    h : ∀ (c : ι), Membership.mem s c → IsSquare (f c)
    i : Subtype fun x => Membership.mem s x
    ⊢ Eq (f ↑i) (HPow.hPow ⋯.choose 2)
  -/
  exact ((isSquare_iff_exists_sq _).mp (h _ i.2)).choose_spec
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_sdiff_eq_prod_sdiff_iff :
    ∏ i ∈ s \ t, f i = ∏ i ∈ t \ s, f i ↔ ∏ i ∈ s, f i = ∏ i ∈ t, f i :=
  eq_comm.trans <| eq_iff_eq_of_mul_eq_mul <| by
    rw [← prod_union disjoint_sdiff_self_left, ← prod_union disjoint_sdiff_self_left,
      sdiff_union_self_eq_union, sdiff_union_self_eq_union, union_comm]


@[to_additive]
lemma prod_sdiff_ne_prod_sdiff_iff :
    ∏ i ∈ s \ t, f i ≠ ∏ i ∈ t \ s, f i ↔ ∏ i ∈ s, f i ≠ ∏ i ∈ t, f i :=
  prod_sdiff_eq_prod_sdiff_iff.not


                                                                /-
                                                                  α : Type u_3
                                                                  s : Finset α
                                                                  ⊢ Eq s.card (s.sum fun x => 1)
                                                                -/
theorem card_eq_sum_ones (s : Finset α) : #s = ∑ _ ∈ s, 1 := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem sum_const_nat {m : ℕ} {f : α → ℕ} (h₁ : ∀ x ∈ s, f x = m) : ∑ x ∈ s, f x = #s * m := by
  /-
    α : Type u_3
    s : Finset α
    m : Nat
    f : α → Nat
    h₁ : ∀ (x : α), Membership.mem s x → Eq (f x) m
    ⊢ Eq (s.sum fun x => f x) (HMul.hMul s.card m)
  -/
  rw [← Nat.nsmul_eq_mul, ← sum_const]
  /-
    α : Type u_3
    s : Finset α
    m : Nat
    f : α → Nat
    h₁ : ∀ (x : α), Membership.mem s x → Eq (f x) m
    ⊢ Eq (s.sum fun x => f x) (s.sum fun _x => m)
  -/
  apply sum_congr rfl h₁
  /-
    🎉 no goals
  -/


lemma sum_card_fiberwise_eq_card_filter {κ : Type*} [DecidableEq κ] (s : Finset ι) (t : Finset κ)
    (g : ι → κ) : ∑ j ∈ t, #{i ∈ s | g i = j} = #{i ∈ s | g i ∈ t} := by
  /-
    ι : Type u_1
    κ : Type u_6
    inst✝ : DecidableEq κ
    s : Finset ι
    t : Finset κ
    g : ι → κ
    ⊢ Eq (t.sum fun j => (Finset.filter (fun i => Eq (g i) j) s).card) (Finset.fil …
  -/
  simpa only [card_eq_sum_ones] using sum_fiberwise_eq_sum_filter _ _ _ _
  /-
    🎉 no goals
  -/


lemma card_filter (p) [DecidablePred p] (s : Finset ι) :
                                                  /-
                                                    ι : Type u_1
                                                    p : ι → Prop
                                                    inst✝ : DecidablePred p
                                                    s : Finset ι
                                                    ⊢ Eq (Finset.filter (fun i => p i) s).card (s.sum fun i => ite (p i) 1 0)
                                                  -/
    #{i ∈ s | p i} = ∑ i ∈ s, ite (p i) 1 0 := by simp [sum_ite]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Moving to the opposite additive commutative monoid commutes with summing. -/
@[simp]
theorem op_sum [AddCommMonoid β] {s : Finset α} (f : α → β) :
    op (∑ x ∈ s, f x) = ∑ x ∈ s, op (f x) :=
  map_sum (opAddEquiv : β ≃+ βᵐᵒᵖ) _ _


@[simp]
theorem unop_sum [AddCommMonoid β] {s : Finset α} (f : α → βᵐᵒᵖ) :
    unop (∑ x ∈ s, f x) = ∑ x ∈ s, unop (f x) :=
  map_sum (opAddEquiv : β ≃+ βᵐᵒᵖ).symm _ _


@[to_additive (attr := simp)]
theorem prod_inv_distrib : (∏ x ∈ s, (f x)⁻¹) = (∏ x ∈ s, f x)⁻¹ :=
  Multiset.prod_map_inv


@[to_additive (attr := simp)]
theorem prod_div_distrib : ∏ x ∈ s, f x / g x = (∏ x ∈ s, f x) / ∏ x ∈ s, g x :=
  Multiset.prod_map_div


@[to_additive]
theorem prod_zpow (f : α → β) (s : Finset α) (n : ℤ) : ∏ a ∈ s, f a ^ n = (∏ a ∈ s, f a) ^ n :=
  Multiset.prod_map_zpow


@[to_additive (attr := simp)]
theorem prod_sdiff_eq_div (h : s₁ ⊆ s₂) :
    ∏ x ∈ s₂ \ s₁, f x = (∏ x ∈ s₂, f x) / ∏ x ∈ s₁, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝¹ : CommGroup β
    inst✝ : DecidableEq α
    h : HasSubset.Subset s₁ s₂
    ⊢ Eq ((SDiff.sdiff s₂ s₁).prod fun x => f x) (HDiv.hDiv (s₂.prod fun x => f x) …
  -/
  rw [eq_div_iff_mul_eq', prod_sdiff h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_sdiff_div_prod_sdiff :
    (∏ x ∈ s₂ \ s₁, f x) / ∏ x ∈ s₁ \ s₂, f x = (∏ x ∈ s₂, f x) / ∏ x ∈ s₁, f x := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    f : α → β
    inst✝¹ : CommGroup β
    inst✝ : DecidableEq α
    ⊢ Eq (HDiv.hDiv ((SDiff.sdiff s₂ s₁).prod fun x => f x) ((SDiff.sdiff s₁ s₂).p …
  -/
  simp [← Finset.prod_sdiff (@inf_le_left _ _ s₁ s₂), ← Finset.prod_sdiff (@inf_le_right _ _ s₁ s₂)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_erase_eq_div {a : α} (h : a ∈ s) :
    ∏ x ∈ s.erase a, f x = (∏ x ∈ s, f x) / f a := by
  /-
    α : Type u_3
    β : Type u_4
    s : Finset α
    f : α → β
    inst✝¹ : CommGroup β
    inst✝ : DecidableEq α
    a : α
    h : Membership.mem s a
    ⊢ Eq ((s.erase a).prod fun x => f x) (HDiv.hDiv (s.prod fun x => f x) (f a))
  -/
  rw [eq_div_iff_mul_eq', prod_erase_mul _ _ h]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_sigma {σ : α → Type*} (s : Finset α) (t : ∀ a, Finset (σ a)) :
    #(s.sigma t) = ∑ a ∈ s, #(t a) :=
  Multiset.card_sigma _ _


@[simp]
theorem card_disjiUnion (s : Finset α) (t : α → Finset β) (h) :
    #(s.disjiUnion t h) = ∑ a ∈ s, #(t a) :=
  Multiset.card_bind _ _


theorem card_biUnion [DecidableEq β] {s : Finset α} {t : α → Finset β}
    (h : ∀ x ∈ s, ∀ y ∈ s, x ≠ y → Disjoint (t x) (t y)) : #(s.biUnion t) = ∑ u ∈ s, #(t u) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : DecidableEq β
    s : Finset α
    t : α → Finset β
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Ne x y → D …
    ⊢ Eq (s.biUnion t).card (s.sum fun u => (t u).card)
  -/
  simpa using sum_biUnion h (β := ℕ) (f := 1)
  /-
    🎉 no goals
  -/


theorem card_biUnion_le [DecidableEq β] {s : Finset α} {t : α → Finset β} :
    #(s.biUnion t) ≤ ∑ a ∈ s, #(t a) :=
  haveI := Classical.decEq α
                            /-
                              α : Type u_3
                              β : Type u_4
                              inst✝ : DecidableEq β
                              s : Finset α
                              t : α → Finset β
                              this : DecidableEq α
                              ⊢ LE.le (EmptyCollection.emptyCollection.biUnion t).card (EmptyCollection.empt …
                            -/
  Finset.induction_on s (by simp) fun a s has ih =>
                            /-
                              🎉 no goals
                            -/
    calc
      #((insert a s).biUnion t) ≤ #(t a) + #(s.biUnion t) := by
        /-
          α : Type u_3
          β : Type u_4
          inst✝ : DecidableEq β
          s✝ : Finset α
          t : α → Finset β
          this : DecidableEq α
          a : α
          s : Finset α
          has : Not (Membership.mem s a)
          ih : LE.le (s.biUnion t).card (s.sum fun a => (t a).card)
          ⊢ LE.le ((Insert.insert a s).biUnion t).card (HAdd.hAdd (t a).card (s.biUnion  …
        -/
        rw [biUnion_insert]; exact card_union_le ..
                             /-
                               🎉 no goals
                             -/
                                         /-
                                           α : Type u_3
                                           β : Type u_4
                                           inst✝ : DecidableEq β
                                           s✝ : Finset α
                                           t : α → Finset β
                                           this : DecidableEq α
                                           a : α
                                           s : Finset α
                                           has : Not (Membership.mem s a)
                                           ih : LE.le (s.biUnion t).card (s.sum fun a => (t a).card)
                                           ⊢ LE.le (HAdd.hAdd (t a).card (s.biUnion t).card) ((Insert.insert a s).sum fun …
                                         -/
      _ ≤ ∑ a ∈ insert a s, #(t a) := by rw [sum_insert has]; exact Nat.add_le_add_left ih _
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem card_eq_sum_card_fiberwise [DecidableEq β] {f : α → β} {s : Finset α} {t : Finset β}
    (H : ∀ x ∈ s, f x ∈ t) : #s = ∑ b ∈ t, #{a ∈ s | f a = b} := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    t : Finset β
    H : ∀ (x : α), Membership.mem s x → Membership.mem t (f x)
    ⊢ Eq s.card (t.sum fun b => (Finset.filter (fun a => Eq (f a) b) s).card)
  -/
  simp only [card_eq_sum_ones, sum_fiberwise_of_maps_to H]
  /-
    🎉 no goals
  -/


theorem card_eq_sum_card_image [DecidableEq β] (f : α → β) (s : Finset α) :
    #s = ∑ b ∈ s.image f, #{a ∈ s | f a = b} :=
  card_eq_sum_card_fiberwise fun _ => mem_image_of_mem _


theorem mem_sum {f : α → Multiset β} (s : Finset α) (b : β) :
    (b ∈ ∑ x ∈ s, f x) ↔ ∃ a ∈ s, b ∈ f a := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a t hi ih => simp [sum_cons, ih, or_and_right, exists_or]


@[to_additive]
theorem prod_unique_nonempty {α β : Type*} [CommMonoid β] [Unique α] (s : Finset α) (f : α → β)
    (h : s.Nonempty) : ∏ x ∈ s, f x = f default := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : CommMonoid β
    inst✝ : Unique α
    s : Finset α
    f : α → β
    h : s.Nonempty
    ⊢ Eq (s.prod fun x => f x) (f Inhabited.default)
  -/
  rw [h.eq_singleton_default, Finset.prod_singleton]
  /-
    🎉 no goals
  -/


theorem sum_nat_mod (s : Finset α) (n : ℕ) (f : α → ℕ) :
    (∑ i ∈ s, f i) % n = (∑ i ∈ s, f i % n) % n :=
                                         /-
                                           α : Type u_3
                                           s : Finset α
                                           n : Nat
                                           f : α → Nat
                                           ⊢ Eq (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) (Multiset.map (fun i => …
                                         -/
  (Multiset.sum_nat_mod _ _).trans <| by rw [Finset.sum, Multiset.map_map]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem prod_nat_mod (s : Finset α) (n : ℕ) (f : α → ℕ) :
    (∏ i ∈ s, f i) % n = (∏ i ∈ s, f i % n) % n :=
                                          /-
                                            α : Type u_3
                                            s : Finset α
                                            n : Nat
                                            f : α → Nat
                                            ⊢ Eq (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) (Multiset.map (fun i => …
                                          -/
  (Multiset.prod_nat_mod _ _).trans <| by rw [Finset.prod, Multiset.map_map]; rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem sum_int_mod (s : Finset α) (n : ℤ) (f : α → ℤ) :
    (∑ i ∈ s, f i) % n = (∑ i ∈ s, f i % n) % n :=
                                         /-
                                           α : Type u_3
                                           s : Finset α
                                           n : Int
                                           f : α → Int
                                           ⊢ Eq (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) (Multiset.map (fun i => …
                                         -/
  (Multiset.sum_int_mod _ _).trans <| by rw [Finset.sum, Multiset.map_map]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem prod_int_mod (s : Finset α) (n : ℤ) (f : α → ℤ) :
    (∏ i ∈ s, f i) % n = (∏ i ∈ s, f i % n) % n :=
                                          /-
                                            α : Type u_3
                                            s : Finset α
                                            n : Int
                                            f : α → Int
                                            ⊢ Eq (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) (Multiset.map (fun i => …
                                          -/
  (Multiset.prod_int_mod _ _).trans <| by rw [Finset.prod, Multiset.map_map]; rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- `Fintype.prod_bijective` is a variant of `Finset.prod_bij` that accepts `Function.Bijective`.

See `Function.Bijective.prod_comp` for a version without `h`. -/
@[to_additive "`Fintype.sum_bijective` is a variant of `Finset.sum_bij` that accepts
`Function.Bijective`.

See `Function.Bijective.sum_comp` for a version without `h`. "]
lemma prod_bijective (e : ι → κ) (he : e.Bijective) (f : ι → α) (g : κ → α)
    (h : ∀ x, f x = g (e x)) : ∏ x, f x = ∏ x, g x :=
                                     /-
                                       ι : Type u_6
                                       κ : Type u_7
                                       α : Type u_8
                                       inst✝² : Fintype ι
                                       inst✝¹ : Fintype κ
                                       inst✝ : CommMonoid α
                                       e : ι → κ
                                       he : Function.Bijective e
                                       f : ι → α
                                       g : κ → α
                                       h : ∀ (x : ι), Eq (f x) (g (e x))
                                       ⊢ ∀ (i : ι), Iff (Membership.mem Finset.univ i) (Membership.mem Finset.univ (( …
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  prod_equiv (.ofBijective e he) (by simp) (by simp [h])
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive] alias _root_.Function.Bijective.finset_prod := prod_bijective


/-- `Fintype.prod_equiv` is a specialization of `Finset.prod_bij` that
automatically fills in most arguments.

See `Equiv.prod_comp` for a version without `h`.
-/
@[to_additive "`Fintype.sum_equiv` is a specialization of `Finset.sum_bij` that
automatically fills in most arguments.

See `Equiv.sum_comp` for a version without `h`."]
lemma prod_equiv (e : ι ≃ κ) (f : ι → α) (g : κ → α) (h : ∀ x, f x = g (e x)) :
    ∏ x, f x = ∏ x, g x := prod_bijective _ e.bijective _ _ h


@[to_additive]
lemma _root_.Function.Bijective.prod_comp {e : ι → κ} (he : e.Bijective) (g : κ → α) :
    ∏ i, g (e i) = ∏ i, g i := prod_bijective _ he _ _ fun _ ↦ rfl


@[to_additive]
lemma _root_.Equiv.prod_comp (e : ι ≃ κ) (g : κ → α) : ∏ i, g (e i) = ∏ i, g i :=
  prod_equiv e _ _ fun _ ↦ rfl


@[to_additive]
lemma prod_of_injective (e : ι → κ) (he : Injective e) (f : ι → α) (g : κ → α)
    (h' : ∀ i ∉ Set.range e, g i = 1) (h : ∀ i, f i = g (e i)) : ∏ i, f i = ∏ j, g j :=
                               /-
                                 ι : Type u_6
                                 κ : Type u_7
                                 α : Type u_8
                                 inst✝² : Fintype ι
                                 inst✝¹ : Fintype κ
                                 inst✝ : CommMonoid α
                                 e : ι → κ
                                 he : Function.Injective e
                                 f : ι → α
                                 g : κ → α
                                 h' : ∀ (i : κ), Not (Membership.mem (Set.range e) i) → Eq (g i) 1
                                 h : ∀ (i : ι), Eq (f i) (g (e i))
                                 ⊢ Set.MapsTo e ↑Finset.univ ↑Finset.univ
                               -/
                               /-
                                 🎉 no goals
                               -/
  prod_of_injOn e he.injOn (by simp) (by simpa using h') (fun i _ ↦ h i)
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive]
lemma prod_fiberwise [DecidableEq κ] (g : ι → κ) (f : ι → α) :
    ∏ j, ∏ i : {i // g i = j}, f i = ∏ i, f i := by
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝³ : Fintype ι
    inst✝² : Fintype κ
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    g : ι → κ
    f : ι → α
    ⊢ Eq (Finset.univ.prod fun j => Finset.univ.prod fun i => f ↑i) (Finset.univ.p …
  -/
  rw [← Finset.prod_fiberwise _ g f]
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝³ : Fintype ι
    inst✝² : Fintype κ
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    g : ι → κ
    f : ι → α
    ⊢ Eq (Finset.univ.prod fun j => Finset.univ.prod fun i => f ↑i) (Finset.univ.p …
  -/
  congr with j
  /-
    case e_f.h
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝³ : Fintype ι
    inst✝² : Fintype κ
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    g : ι → κ
    f : ι → α
    j : κ
    ⊢ Eq (Finset.univ.prod fun i => f ↑i) ((Finset.filter (fun i => Eq (g i) j) Fi …
  -/
  exact (prod_subtype _ (by simp) _).symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_fiberwise' [DecidableEq κ] (g : ι → κ) (f : κ → α) :
    ∏ j, ∏ _i : {i // g i = j}, f j = ∏ i, f (g i) := by
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝³ : Fintype ι
    inst✝² : Fintype κ
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    g : ι → κ
    f : κ → α
    ⊢ Eq (Finset.univ.prod fun j => Finset.univ.prod fun _i => f j) (Finset.univ.p …
  -/
  rw [← Finset.prod_fiberwise' _ g f]
  /-
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝³ : Fintype ι
    inst✝² : Fintype κ
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    g : ι → κ
    f : κ → α
    ⊢ Eq (Finset.univ.prod fun j => Finset.univ.prod fun _i => f j) (Finset.univ.p …
  -/
  congr with j
  /-
    case e_f.h
    ι : Type u_6
    κ : Type u_7
    α : Type u_8
    inst✝³ : Fintype ι
    inst✝² : Fintype κ
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq κ
    g : ι → κ
    f : κ → α
    j : κ
    ⊢ Eq (Finset.univ.prod fun _i => f j) ((Finset.filter (fun i => Eq (g i) j) Fi …
  -/
  exact (prod_subtype _ (by simp) fun _ ↦ _).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_unique {α β : Type*} [CommMonoid β] [Unique α] [Fintype α] (f : α → β) :
                                   /-
                                     α : Type u_9
                                     β : Type u_10
                                     inst✝² : CommMonoid β
                                     inst✝¹ : Unique α
                                     inst✝ : Fintype α
                                     f : α → β
                                     ⊢ Eq (Finset.univ.prod fun x => f x) (f Inhabited.default)
                                   -/
    ∏ x : α, f x = f default := by rw [univ_unique, prod_singleton]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
theorem prod_empty {α β : Type*} [CommMonoid β] [IsEmpty α] [Fintype α] (f : α → β) :
    ∏ x : α, f x = 1 :=
  Finset.prod_of_isEmpty _


@[to_additive]
theorem prod_subsingleton {α β : Type*} [CommMonoid β] [Subsingleton α] [Fintype α] (f : α → β)
    (a : α) : ∏ x : α, f x = f a := by
  /-
    α : Type u_9
    β : Type u_10
    inst✝² : CommMonoid β
    inst✝¹ : Subsingleton α
    inst✝ : Fintype α
    f : α → β
    a : α
    ⊢ Eq (Finset.univ.prod fun x => f x) (f a)
  -/
  have : Unique α := uniqueOfSubsingleton a
  /-
    α : Type u_9
    β : Type u_10
    inst✝² : CommMonoid β
    inst✝¹ : Subsingleton α
    inst✝ : Fintype α
    f : α → β
    a : α
    this : Unique α
    ⊢ Eq (Finset.univ.prod fun x => f x) (f a)
  -/
  rw [prod_unique f, Subsingleton.elim default a]
  /-
    🎉 no goals
  -/


@[to_additive] theorem prod_Prop {β} [CommMonoid β] (f : Prop → β) :
                                      /-
                                        β : Type u_9
                                        inst✝ : CommMonoid β
                                        f : Prop → β
                                        ⊢ Eq (Finset.univ.prod fun p => f p) (HMul.hMul (f True) (f False))
                                      -/
    ∏ p, f p = f True * f False := by simp
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
theorem prod_subtype_mul_prod_subtype {α β : Type*} [Fintype α] [CommMonoid β] (p : α → Prop)
    (f : α → β) [DecidablePred p] :
    (∏ i : { x // p x }, f i) * ∏ i : { x // ¬p x }, f i = ∏ i, f i := by
  classical
    let s := { x | p x }.toFinset
    rw [← Finset.prod_subtype s, ← Finset.prod_subtype sᶜ]
    · exact Finset.prod_mul_prod_compl _ _
    · simp [s]
    · simp [s]


@[to_additive] lemma prod_subset {s : Finset ι} {f : ι → α} (h : ∀ i, f i ≠ 1 → i ∈ s) :
    ∏ i ∈ s, f i = ∏ i, f i :=
                                         /-
                                           ι : Type u_6
                                           α : Type u_8
                                           inst✝¹ : Fintype ι
                                           inst✝ : CommMonoid α
                                           s : Finset ι
                                           f : ι → α
                                           h : ∀ (i : ι), Ne (f i) 1 → Membership.mem s i
                                           ⊢ ∀ (x : ι), Membership.mem Finset.univ x → Not (Membership.mem s x) → Eq (f x …
                                         -/
  Finset.prod_subset s.subset_univ <| by simpa [not_imp_comm (a := _ ∈ s)]
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive]
lemma prod_ite_eq_ite_exists (p : ι → Prop) [DecidablePred p] (h : ∀ i j, p i → p j → i = j)
    (a : α) : ∏ i, ite (p i) a 1 = ite (∃ i, p i) a 1 := by
  /-
    ι : Type u_6
    α : Type u_8
    inst✝² : Fintype ι
    inst✝¹ : CommMonoid α
    p : ι → Prop
    inst✝ : DecidablePred p
    h : ∀ (i j : ι), p i → p j → Eq i j
    a : α
    ⊢ Eq (Finset.univ.prod fun i => ite (p i) a 1) (ite (Exists fun i => p i) a 1)
  -/
  simp [prod_ite_one univ p (by simpa using h)]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_ite_mem (s : Finset ι) (f : ι → α) : ∏ i, (if i ∈ s then f i else 1) = ∏ i ∈ s, f i := by
  /-
    ι : Type u_6
    α : Type u_8
    inst✝² : Fintype ι
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    s : Finset ι
    f : ι → α
    ⊢ Eq (Finset.univ.prod fun i => ite (Membership.mem s i) (f i) 1) (s.prod fun  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- See also `Finset.prod_dite_eq`. -/
@[to_additive "See also `Finset.sum_dite_eq`."] lemma prod_dite_eq (i : ι) (f : ∀ j, i = j → α) :
    ∏ j, (if h : i = j then f j h else 1) = f i rfl := by
  /-
    ι : Type u_6
    α : Type u_8
    inst✝² : Fintype ι
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    i : ι
    f : (j : ι) → Eq i j → α
    ⊢ Eq (Finset.univ.prod fun j => dite (Eq i j) (fun h => f j h) fun h => 1) (f  …
  -/
  rw [Finset.prod_dite_eq, if_pos (mem_univ _)]
  /-
    🎉 no goals
  -/


/-- See also `Finset.prod_dite_eq'`. -/
@[to_additive "See also `Finset.sum_dite_eq'`."] lemma prod_dite_eq' (i : ι) (f : ∀ j, j = i → α) :
    ∏ j, (if h : j = i then f j h else 1) = f i rfl := by
  /-
    ι : Type u_6
    α : Type u_8
    inst✝² : Fintype ι
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    i : ι
    f : (j : ι) → Eq j i → α
    ⊢ Eq (Finset.univ.prod fun j => dite (Eq j i) (fun h => f j h) fun h => 1) (f  …
  -/
  rw [Finset.prod_dite_eq', if_pos (mem_univ _)]
  /-
    🎉 no goals
  -/


/-- See also `Finset.prod_ite_eq`. -/
@[to_additive "See also `Finset.sum_ite_eq`."]
lemma prod_ite_eq (i : ι) (f : ι → α) : ∏ j, (if i = j then f j else 1) = f i := by
  /-
    ι : Type u_6
    α : Type u_8
    inst✝² : Fintype ι
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    i : ι
    f : ι → α
    ⊢ Eq (Finset.univ.prod fun j => ite (Eq i j) (f j) 1) (f i)
  -/
  rw [Finset.prod_ite_eq, if_pos (mem_univ _)]
  /-
    🎉 no goals
  -/


/-- See also `Finset.prod_ite_eq'`. -/
@[to_additive "See also `Finset.sum_ite_eq'`."]
lemma prod_ite_eq' (i : ι) (f : ι → α) : ∏ j, (if j = i then f j else 1) = f i := by
  /-
    ι : Type u_6
    α : Type u_8
    inst✝² : Fintype ι
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    i : ι
    f : ι → α
    ⊢ Eq (Finset.univ.prod fun j => ite (Eq j i) (f j) 1) (f i)
  -/
  rw [Finset.prod_ite_eq', if_pos (mem_univ _)]
  /-
    🎉 no goals
  -/


/-- See also `Finset.prod_pi_mulSingle`. -/
@[to_additive "See also `Finset.sum_pi_single`."]
lemma prod_pi_mulSingle {α : ι → Type*} [∀ i, CommMonoid (α i)] (i : ι) (f : ∀ i, α i) :
    ∏ j, Pi.mulSingle j (f j) i = f i := prod_dite_eq _ _


/-- See also `Finset.prod_pi_mulSingle'`. -/
@[to_additive "See also `Finset.sum_pi_single'`."]
lemma prod_pi_mulSingle' (i : ι) (a : α) : ∏ j, Pi.mulSingle i a j = a := prod_dite_eq' _ _


@[to_additive (attr := simp)]
lemma prod_attach_univ [Fintype ι] (f : {i // i ∈ @univ ι _} → α) :
    ∏ i ∈ univ.attach, f i = ∏ i, f ⟨i, mem_univ _⟩ :=
                                                                 /-
                                                                   ι : Type u_1
                                                                   α : Type u_3
                                                                   inst✝¹ : CommMonoid α
                                                                   inst✝ : Fintype ι
                                                                   f : (Subtype fun i => Membership.mem Finset.univ i) → α
                                                                   ⊢ ∀ (x : Subtype (Membership.mem Finset.univ)), Eq (f x) (f ⟨(Equiv.subtypeUni …
                                                                 -/
  Fintype.prod_equiv (Equiv.subtypeUnivEquiv mem_univ) _ _ <| by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
theorem prod_erase_attach [DecidableEq ι] {s : Finset ι} (f : ι → α) (i : ↑s) :
    ∏ j ∈ s.attach.erase i, f ↑j = ∏ j ∈ s.erase ↑i, f j := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    s : Finset ι
    f : ι → α
    i : Subtype fun x => Membership.mem s x
    ⊢ Eq ((s.attach.erase i).prod fun j => f ↑j) ((s.erase ↑i).prod fun j => f j)
  -/
  rw [← Function.Embedding.coe_subtype, ← prod_map]
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : DecidableEq ι
    s : Finset ι
    f : ι → α
    i : Subtype fun x => Membership.mem s x
    ⊢ Eq ((Finset.map (Function.Embedding.subtype fun x => Membership.mem s x) (s. …
  -/
  simp [attach_map_val]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_toFinset {M : Type*} [DecidableEq α] [CommMonoid M] (f : α → M) :
    ∀ {l : List α} (_hl : l.Nodup), l.toFinset.prod f = (l.map f).prod
                /-
                  α : Type u_3
                  M : Type u_6
                  inst✝¹ : DecidableEq α
                  inst✝ : CommMonoid M
                  f : α → M
                  x✝ : List.nil.Nodup
                  ⊢ Eq (List.nil.toFinset.prod f) (List.map f List.nil).prod
                -/
  | [], _ => by simp
                /-
                  🎉 no goals
                -/
  | a :: l, hl => by
    /-
      α : Type u_3
      M : Type u_6
      inst✝¹ : DecidableEq α
      inst✝ : CommMonoid M
      f : α → M
      a : α
      l : List α
      hl : (List.cons a l).Nodup
      ⊢ Eq ((List.cons a l).toFinset.prod f) (List.map f (List.cons a l)).prod
    -/
    let ⟨not_mem, hl⟩ := List.nodup_cons.mp hl
    /-
      α : Type u_3
      M : Type u_6
      inst✝¹ : DecidableEq α
      inst✝ : CommMonoid M
      f : α → M
      a : α
      l : List α
      hl✝ : (List.cons a l).Nodup
      not_mem : Not (Membership.mem l a)
      hl : l.Nodup
      ⊢ Eq ((List.cons a l).toFinset.prod f) (List.map f (List.cons a l)).prod
    -/
    simp [Finset.prod_insert (mt List.mem_toFinset.mp not_mem), prod_toFinset _ hl]
    /-
      🎉 no goals
    -/


@[simp]
theorem sum_toFinset_count_eq_length [DecidableEq α] (l : List α) :
    ∑ a in l.toFinset, l.count a = l.length := by
  /-
    α : Type u_3
    inst✝ : DecidableEq α
    l : List α
    ⊢ Eq (l.toFinset.sum fun a => List.count a l) l.length
  -/
  simpa [List.map_const'] using (Finset.sum_list_map_count l fun _ => (1 : ℕ)).symm
  /-
    🎉 no goals
  -/


@[simp]
lemma card_sum (s : Finset ι) (f : ι → Multiset α) : card (∑ i ∈ s, f i) = ∑ i ∈ s, card (f i) :=
  map_sum cardHom ..


theorem disjoint_list_sum_left {a : Multiset α} {l : List (Multiset α)} :
    Disjoint l.sum a ↔ ∀ b ∈ l, Disjoint b a := by
  induction l with
  | nil =>
    simp only [zero_disjoint, List.not_mem_nil, IsEmpty.forall_iff, forall_const, List.sum_nil]
  | cons b bs ih =>
    simp_rw [List.sum_cons, disjoint_add_left, List.mem_cons, forall_eq_or_imp]
    simp [and_congr_left_iff, ih]


theorem disjoint_list_sum_right {a : Multiset α} {l : List (Multiset α)} :
    Disjoint a l.sum ↔ ∀ b ∈ l, Disjoint a b := by
  /-
    α : Type u_3
    a : Multiset α
    l : List (Multiset α)
    ⊢ Iff (Disjoint a l.sum) (∀ (b : Multiset α), Membership.mem l b → Disjoint a b)
  -/
  simpa only [disjoint_comm (a := a)] using disjoint_list_sum_left
  /-
    🎉 no goals
  -/


theorem disjoint_sum_left {a : Multiset α} {i : Multiset (Multiset α)} :
    Disjoint i.sum a ↔ ∀ b ∈ i, Disjoint b a :=
  Quotient.inductionOn i fun l => by
    /-
      α : Type u_3
      a : Multiset α
      i : Multiset (Multiset α)
      l : List (Multiset α)
      ⊢ Iff (Disjoint (Multiset.sum (Quotient.mk (List.isSetoid (Multiset α)) l)) a) …
    -/
    rw [quot_mk_to_coe, Multiset.sum_coe]
    /-
      α : Type u_3
      a : Multiset α
      i : Multiset (Multiset α)
      l : List (Multiset α)
      ⊢ Iff (Disjoint l.sum a) (∀ (b : Multiset α), Membership.mem (↑l) b → Disjoint …
    -/
    exact disjoint_list_sum_left
    /-
      🎉 no goals
    -/


theorem disjoint_sum_right {a : Multiset α} {i : Multiset (Multiset α)} :
    Disjoint a i.sum ↔ ∀ b ∈ i, Disjoint a b := by
  /-
    α : Type u_3
    a : Multiset α
    i : Multiset (Multiset α)
    ⊢ Iff (Disjoint a i.sum) (∀ (b : Multiset α), Membership.mem i b → Disjoint a b)
  -/
  simpa only [disjoint_comm (a := a)] using disjoint_sum_left
  /-
    🎉 no goals
  -/


theorem disjoint_finset_sum_left {β : Type*} {i : Finset β} {f : β → Multiset α} {a : Multiset α} :
    Disjoint (i.sum f) a ↔ ∀ b ∈ i, Disjoint (f b) a := by
  /-
    α : Type u_3
    β : Type u_6
    i : Finset β
    f : β → Multiset α
    a : Multiset α
    ⊢ Iff (Disjoint (i.sum f) a) (∀ (b : β), Membership.mem i b → Disjoint (f b) a)
  -/
  convert @disjoint_sum_left _ a (map f i.val)
  /-
    case h.e'_2.a
    α : Type u_3
    β : Type u_6
    i : Finset β
    f : β → Multiset α
    a : Multiset α
    ⊢ Iff (∀ (b : β), Membership.mem i b → Disjoint (f b) a) (∀ (b : Multiset α),  …
  -/
  simp [and_congr_left_iff]
  /-
    🎉 no goals
  -/


theorem disjoint_finset_sum_right {β : Type*} {i : Finset β} {f : β → Multiset α}
    {a : Multiset α} : Disjoint a (i.sum f) ↔ ∀ b ∈ i, Disjoint a (f b) := by
  /-
    α : Type u_3
    β : Type u_6
    i : Finset β
    f : β → Multiset α
    a : Multiset α
    ⊢ Iff (Disjoint a (i.sum f)) (∀ (b : β), Membership.mem i b → Disjoint a (f b))
  -/
  simpa only [disjoint_comm] using disjoint_finset_sum_left
  /-
    🎉 no goals
  -/


@[simp]
lemma mem_sum {s : Finset ι} {m : ι → Multiset α} : a ∈ ∑ i ∈ s, m i ↔ ∃ i ∈ s, a ∈ m i := by
  /-
    ι : Type u_1
    α : Type u_3
    a : α
    s : Finset ι
    m : ι → Multiset α
    ⊢ Iff (Membership.mem (s.sum fun i => m i) a) (Exists fun i => And (Membership …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction s using Finset.cons_induction <;> simp [*]
                                              /-
                                                🎉 no goals
                                              -/


theorem toFinset_sum_count_eq (s : Multiset α) : ∑ a in s.toFinset, s.count a = card s := by
  /-
    α : Type u_3
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq (s.toFinset.sum fun a => Multiset.count a s) s.card
  -/
  simpa using (Finset.sum_multiset_map_count s (fun _ => (1 : ℕ))).symm
  /-
    🎉 no goals
  -/


@[simp] lemma sum_count_eq_card {s : Finset α} {m : Multiset α} (hms : ∀ a ∈ m, a ∈ s) :
    ∑ a ∈ s, m.count a = card m := by
  /-
    α : Type u_3
    inst✝ : DecidableEq α
    s : Finset α
    m : Multiset α
    hms : ∀ (a : α), Membership.mem m a → Membership.mem s a
    ⊢ Eq (s.sum fun a => Multiset.count a m) m.card
  -/
  rw [← toFinset_sum_count_eq, ← Finset.sum_filter_ne_zero]
  /-
    α : Type u_3
    inst✝ : DecidableEq α
    s : Finset α
    m : Multiset α
    hms : ∀ (a : α), Membership.mem m a → Membership.mem s a
    ⊢ Eq ((Finset.filter (fun x => Ne (Multiset.count x m) 0) s).sum fun x => Mult …
  -/
  congr with a
  /-
    case e_s.h
    α : Type u_3
    inst✝ : DecidableEq α
    s : Finset α
    m : Multiset α
    hms : ∀ (a : α), Membership.mem m a → Membership.mem s a
    a : α
    ⊢ Iff (Membership.mem (Finset.filter (fun x => Ne (Multiset.count x m) 0) s) a …
  -/
  simpa using hms a
  /-
    🎉 no goals
  -/


@[deprecated sum_count_eq_card (since := "2024-07-21")]
                                                                                           /-
                                                                                             α : Type u_3
                                                                                             inst✝¹ : DecidableEq α
                                                                                             inst✝ : Fintype α
                                                                                             s : Multiset α
                                                                                             ⊢ Eq (Finset.univ.sum fun a => Multiset.count a s) s.card
                                                                                           -/
theorem sum_count_eq [Fintype α] (s : Multiset α) : ∑ a, s.count a = Multiset.card s := by simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem count_sum' {s : Finset β} {a : α} {f : β → Multiset α} :
    count a (∑ x ∈ s, f x) = ∑ x ∈ s, count a (f x) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : DecidableEq α
    s : Finset β
    a : α
    f : β → Multiset α
    ⊢ Eq (Multiset.count a (s.sum fun x => f x)) (s.sum fun x => Multiset.count a  …
  -/
  dsimp only [Finset.sum]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : DecidableEq α
    s : Finset β
    a : α
    f : β → Multiset α
    ⊢ Eq (Multiset.count a (Multiset.map (fun x => f x) s.val).sum) (Multiset.map  …
  -/
  rw [count_sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_sum_count_nsmul_eq (s : Multiset α) :
    ∑ a ∈ s.toFinset, s.count a • {a} = s := by
  /-
    α : Type u_3
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq (s.toFinset.sum fun a => HSMul.hSMul (Multiset.count a s) (Singleton.sing …
  -/
  rw [← Finset.sum_multiset_map_count, Multiset.sum_map_singleton]
  /-
    🎉 no goals
  -/


theorem exists_smul_of_dvd_count (s : Multiset α) {k : ℕ}
    (h : ∀ a : α, a ∈ s → k ∣ Multiset.count a s) : ∃ u : Multiset α, s = k • u := by
  /-
    α : Type u_3
    inst✝ : DecidableEq α
    s : Multiset α
    k : Nat
    h : ∀ (a : α), Membership.mem s a → Dvd.dvd k (Multiset.count a s)
    ⊢ Exists fun u => Eq s (HSMul.hSMul k u)
  -/
  use ∑ a ∈ s.toFinset, (s.count a / k) • {a}
  have h₂ :
    (∑ x ∈ s.toFinset, k • (count x s / k) • ({x} : Multiset α)) =
      ∑ x ∈ s.toFinset, count x s • {x} := by
    apply Finset.sum_congr rfl
    intro x hx
    rw [← mul_nsmul', Nat.mul_div_cancel' (h x (mem_toFinset.mp hx))]
  /-
    case h
    α : Type u_3
    inst✝ : DecidableEq α
    s : Multiset α
    k : Nat
    h : ∀ (a : α), Membership.mem s a → Dvd.dvd k (Multiset.count a s)
    h₂ : Eq (s.toFinset.sum fun x => HSMul.hSMul k (HSMul.hSMul (HDiv.hDiv (Multis …
    ⊢ Eq s (HSMul.hSMul k (s.toFinset.sum fun a => HSMul.hSMul (HDiv.hDiv (Multise …
  -/
  rw [← Finset.sum_nsmul, h₂, toFinset_sum_count_nsmul_eq]
  /-
    🎉 no goals
  -/


theorem toFinset_prod_dvd_prod [CommMonoid α] (S : Multiset α) : S.toFinset.prod id ∣ S.prod := by
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    S : Multiset α
    ⊢ Dvd.dvd (S.toFinset.prod id) S.prod
  -/
  rw [Finset.prod_eq_multiset_prod]
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    S : Multiset α
    ⊢ Dvd.dvd (Multiset.map id S.toFinset.val).prod S.prod
  -/
  refine Multiset.prod_dvd_prod_of_le ?_
  /-
    α : Type u_3
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    S : Multiset α
    ⊢ LE.le (Multiset.map id S.toFinset.val) S
  -/
  simp [Multiset.dedup_le S]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_sum {α : Type*} {ι : Type*} [CommMonoid α] (f : ι → Multiset α) (s : Finset ι) :
    (∑ x ∈ s, f x).prod = ∏ x ∈ s, (f x).prod := by
  induction s using Finset.cons_induction with
  | empty => rw [Finset.sum_empty, Finset.prod_empty, Multiset.prod_zero]
  | cons a s has ih => rw [Finset.sum_cons, Finset.prod_cons, Multiset.prod_add, ih]


@[simp, norm_cast]
theorem Units.coe_prod {M : Type*} [CommMonoid M] (f : α → Mˣ) (s : Finset α) :
    (↑(∏ i ∈ s, f i) : M) = ∏ i ∈ s, (f i : M) :=
  map_prod (Units.coeHom M) _ _


@[to_additive (attr := simp)]
lemma IsUnit.prod_iff [CommMonoid β] : IsUnit (∏ a ∈ s, f a) ↔ ∀ a ∈ s, IsUnit (f a) := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a s ha hs => rw [Finset.prod_cons, IsUnit.mul_iff, hs, Finset.forall_mem_cons]


@[to_additive]
lemma IsUnit.prod_univ_iff [Fintype α] [CommMonoid β] : IsUnit (∏ a, f a) ↔ ∀ a, IsUnit (f a) := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β
    inst✝¹ : Fintype α
    inst✝ : CommMonoid β
    ⊢ Iff (IsUnit (Finset.univ.prod fun a => f a)) (∀ (a : α), IsUnit (f a))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nat_abs_sum_le {ι : Type*} (s : Finset ι) (f : ι → ℤ) :
    (∑ i ∈ s, f i).natAbs ≤ ∑ i ∈ s, (f i).natAbs := by
  induction s using Finset.cons_induction with
  | empty => simp only [Finset.sum_empty, Int.natAbs_zero, le_refl]
  | cons i s his IH =>
    simp only [Finset.sum_cons, not_false_iff]
    exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add_left IH _)


@[simp]
                                                                              /-
                                                                                α : Type u_3
                                                                                inst✝ : Monoid α
                                                                                s : List α
                                                                                ⊢ Eq (Additive.ofMul s.prod) (List.map (⇑Additive.ofMul) s).sum
                                                                              -/
theorem ofMul_list_prod (s : List α) : ofMul s.prod = (s.map ofMul).sum := by simp [ofMul]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
theorem toMul_list_sum (s : List (Additive α)) : s.sum.toMul = (s.map toMul).prod := by
  /-
    α : Type u_3
    inst✝ : Monoid α
    s : List (Additive α)
    ⊢ Eq (Additive.toMul s.sum) (List.map (⇑Additive.toMul) s).prod
  -/
  simp [toMul, ofMul]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
                                                                              /-
                                                                                α : Type u_3
                                                                                inst✝ : AddMonoid α
                                                                                s : List α
                                                                                ⊢ Eq (Multiplicative.ofAdd s.sum) (List.map (⇑Multiplicative.ofAdd) s).prod
                                                                              -/
theorem ofAdd_list_prod (s : List α) : ofAdd s.sum = (s.map ofAdd).prod := by simp [ofAdd]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
theorem toAdd_list_sum (s : List (Multiplicative α)) : s.prod.toAdd = (s.map toAdd).sum := by
  /-
    α : Type u_3
    inst✝ : AddMonoid α
    s : List (Multiplicative α)
    ⊢ Eq (Multiplicative.toAdd s.prod) (List.map (⇑Multiplicative.toAdd) s).sum
  -/
  simp [toAdd, ofAdd]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem ofMul_multiset_prod (s : Multiset α) : ofMul s.prod = (s.map ofMul).sum := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s : Multiset α
    ⊢ Eq (Additive.ofMul s.prod) (Multiset.map (⇑Additive.ofMul) s).sum
  -/
  simp [ofMul]; rfl
                /-
                  🎉 no goals
                -/


@[simp]
theorem toMul_multiset_sum (s : Multiset (Additive α)) : s.sum.toMul = (s.map toMul).prod := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s : Multiset (Additive α)
    ⊢ Eq (Additive.toMul s.sum) (Multiset.map (⇑Additive.toMul) s).prod
  -/
  simp [toMul, ofMul]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem ofMul_prod (s : Finset ι) (f : ι → α) : ofMul (∏ i ∈ s, f i) = ∑ i ∈ s, ofMul (f i) :=
  rfl


@[simp]
theorem toMul_sum (s : Finset ι) (f : ι → Additive α) :
    (∑ i ∈ s, f i).toMul = ∏ i ∈ s, (f i).toMul :=
  rfl


@[simp]
theorem ofAdd_multiset_prod (s : Multiset α) : ofAdd s.sum = (s.map ofAdd).prod := by
  /-
    α : Type u_3
    inst✝ : AddCommMonoid α
    s : Multiset α
    ⊢ Eq (Multiplicative.ofAdd s.sum) (Multiset.map (⇑Multiplicative.ofAdd) s).prod
  -/
  simp [ofAdd]; rfl
                /-
                  🎉 no goals
                -/


@[simp]
theorem toAdd_multiset_sum (s : Multiset (Multiplicative α)) :
    s.prod.toAdd = (s.map toAdd).sum := by
  /-
    α : Type u_3
    inst✝ : AddCommMonoid α
    s : Multiset (Multiplicative α)
    ⊢ Eq (Multiplicative.toAdd s.prod) (Multiset.map (⇑Multiplicative.toAdd) s).sum
  -/
  simp [toAdd, ofAdd]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem ofAdd_sum (s : Finset ι) (f : ι → α) : ofAdd (∑ i ∈ s, f i) = ∏ i ∈ s, ofAdd (f i) :=
  rfl


@[simp]
theorem toAdd_prod (s : Finset ι) (f : ι → Multiplicative α) :
    (∏ i ∈ s, f i).toAdd = ∑ i ∈ s, (f i).toAdd :=
  rfl


