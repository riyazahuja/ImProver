open Classical in
/-- Sum of `f x` as `x` ranges over the elements of the support of `f`, if it's finite. Zero
otherwise. -/
noncomputable irreducible_def finsum (lemma := finsum_def') [AddCommMonoid M] (f : α → M) : M :=
  if h : (support (f ∘ PLift.down)).Finite then ∑ i ∈ h.toFinset, f i.down else 0


open Classical in
/-- Product of `f x` as `x` ranges over the elements of the multiplicative support of `f`, if it's
finite. One otherwise. -/
@[to_additive existing]
noncomputable irreducible_def finprod (lemma := finprod_def') (f : α → M) : M :=
  if h : (mulSupport (f ∘ PLift.down)).Finite then ∏ i ∈ h.toFinset, f i.down else 1


/-- `∑ᶠ x, f x` is notation for `finsum f`. It is the sum of `f x`, where `x` ranges over the
support of `f`, if it's finite, zero otherwise. Taking the sum over multiple arguments or
conditions is possible, e.g. `∏ᶠ (x) (y), f x y` and `∏ᶠ (x) (h: x ∈ s), f x`-/
notation3"∑ᶠ "(...)", "r:67:(scoped f => finsum f) => r


/-- `∏ᶠ x, f x` is notation for `finprod f`. It is the product of `f x`, where `x` ranges over the
multiplicative support of `f`, if it's finite, one otherwise. Taking the product over multiple
arguments or conditions is possible, e.g. `∏ᶠ (x) (y), f x y` and `∏ᶠ (x) (h: x ∈ s), f x`-/
notation3"∏ᶠ "(...)", "r:67:(scoped f => finprod f) => r

-- Porting note: The following ports the lean3 notation for this file, but is currently very fickle.

-- syntax (name := bigfinsum) "∑ᶠ" extBinders ", " term:67 : term
-- macro_rules (kind := bigfinsum)
--   | `(∑ᶠ $x:ident, $p) => `(finsum (fun $x:ident ↦ $p))
--   | `(∑ᶠ $x:ident : $t, $p) => `(finsum (fun $x:ident : $t ↦ $p))
--   | `(∑ᶠ $x:ident $b:binderPred, $p) =>
--     `(finsum fun $x => (finsum (α := satisfies_binder_pred% $x $b) (fun _ => $p)))

--   | `(∑ᶠ ($x:ident) ($h:ident : $t), $p) =>
--       `(finsum fun ($x) => finsum (α := $t) (fun $h => $p))
--   | `(∑ᶠ ($x:ident : $_) ($h:ident : $t), $p) =>
--       `(finsum fun ($x) => finsum (α := $t) (fun $h => $p))

--   | `(∑ᶠ ($x:ident) ($y:ident), $p) =>
--       `(finsum fun $x => (finsum fun $y => $p))
--   | `(∑ᶠ ($x:ident) ($y:ident) ($h:ident : $t), $p) =>
--       `(finsum fun $x => (finsum fun $y => (finsum (α := $t) fun $h => $p)))

--   | `(∑ᶠ ($x:ident) ($y:ident) ($z:ident), $p) =>
--       `(finsum fun $x => (finsum fun $y => (finsum fun $z => $p)))
--   | `(∑ᶠ ($x:ident) ($y:ident) ($z:ident) ($h:ident : $t), $p) =>
--       `(finsum fun $x => (finsum fun $y => (finsum fun $z => (finsum (α := $t) fun $h => $p))))
--
--
-- syntax (name := bigfinprod) "∏ᶠ " extBinders ", " term:67 : term
-- macro_rules (kind := bigfinprod)
--   | `(∏ᶠ $x:ident, $p) => `(finprod (fun $x:ident ↦ $p))
--   | `(∏ᶠ $x:ident : $t, $p) => `(finprod (fun $x:ident : $t ↦ $p))
--   | `(∏ᶠ $x:ident $b:binderPred, $p) =>
--     `(finprod fun $x => (finprod (α := satisfies_binder_pred% $x $b) (fun _ => $p)))

--   | `(∏ᶠ ($x:ident) ($h:ident : $t), $p) =>
--       `(finprod fun ($x) => finprod (α := $t) (fun $h => $p))
--   | `(∏ᶠ ($x:ident : $_) ($h:ident : $t), $p) =>
--       `(finprod fun ($x) => finprod (α := $t) (fun $h => $p))

--   | `(∏ᶠ ($x:ident) ($y:ident), $p) =>
--       `(finprod fun $x => (finprod fun $y => $p))
--   | `(∏ᶠ ($x:ident) ($y:ident) ($h:ident : $t), $p) =>
--       `(finprod fun $x => (finprod fun $y => (finprod (α := $t) fun $h => $p)))

--   | `(∏ᶠ ($x:ident) ($y:ident) ($z:ident), $p) =>
--       `(finprod fun $x => (finprod fun $y => (finprod fun $z => $p)))
--   | `(∏ᶠ ($x:ident) ($y:ident) ($z:ident) ($h:ident : $t), $p) =>
--       `(finprod fun $x => (finprod fun $y => (finprod fun $z =>
--          (finprod (α := $t) fun $h => $p))))


@[to_additive]
theorem finprod_eq_prod_plift_of_mulSupport_toFinset_subset {f : α → M}
    (hf : (mulSupport (f ∘ PLift.down)).Finite) {s : Finset (PLift α)} (hs : hf.toFinset ⊆ s) :
    ∏ᶠ i, f i = ∏ i ∈ s, f i.down := by
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    hf : (Function.mulSupport (Function.comp f PLift.down)).Finite
    s : Finset (PLift α)
    hs : HasSubset.Subset hf.toFinset s
    ⊢ Eq (finprod fun i => f i) (s.prod fun i => f i.down)
  -/
  rw [finprod, dif_pos]
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    hf : (Function.mulSupport (Function.comp f PLift.down)).Finite
    s : Finset (PLift α)
    hs : HasSubset.Subset hf.toFinset s
    ⊢ Eq ((Set.Finite.toFinset ?hc).prod fun i => f i.down) (s.prod fun i => f i.d …
  -/
  refine Finset.prod_subset hs fun x _ hxf => ?_
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    hf : (Function.mulSupport (Function.comp f PLift.down)).Finite
    s : Finset (PLift α)
    hs : HasSubset.Subset hf.toFinset s
    x : PLift α
    x✝ : Membership.mem s x
    hxf : Not (Membership.mem hf.toFinset x)
    ⊢ Eq (f x.down) 1
  -/
  rwa [hf.mem_toFinset, nmem_mulSupport] at hxf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_eq_prod_plift_of_mulSupport_subset {f : α → M} {s : Finset (PLift α)}
    (hs : mulSupport (f ∘ PLift.down) ⊆ s) : ∏ᶠ i, f i = ∏ i ∈ s, f i.down :=
  finprod_eq_prod_plift_of_mulSupport_toFinset_subset (s.finite_toSet.subset hs) fun x hx => by
    /-
      M : Type u_2
      α : Sort u_4
      inst✝ : CommMonoid M
      f : α → M
      s : Finset (PLift α)
      hs : HasSubset.Subset (Function.mulSupport (Function.comp f PLift.down)) ↑s
      x : PLift α
      hx : Membership.mem ⋯.toFinset x
      ⊢ Membership.mem s x
    -/
    rw [Finite.mem_toFinset] at hx
    /-
      M : Type u_2
      α : Sort u_4
      inst✝ : CommMonoid M
      f : α → M
      s : Finset (PLift α)
      hs : HasSubset.Subset (Function.mulSupport (Function.comp f PLift.down)) ↑s
      x : PLift α
      hx : Membership.mem (Function.mulSupport (Function.comp f PLift.down)) x
      ⊢ Membership.mem s x
    -/
    exact hs hx
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem finprod_one : (∏ᶠ _ : α, (1 : M)) = 1 := by
  have : (mulSupport fun x : PLift α => (fun _ => 1 : α → M) x.down) ⊆ (∅ : Finset (PLift α)) :=
    fun x h => by simp at h
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    this : HasSubset.Subset (Function.mulSupport fun x => (fun x => 1) x.down) ↑Em …
    ⊢ Eq (finprod fun x => 1) 1
  -/
  rw [finprod_eq_prod_plift_of_mulSupport_subset this, Finset.prod_empty]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_of_isEmpty [IsEmpty α] (f : α → M) : ∏ᶠ i, f i = 1 := by
  /-
    M : Type u_2
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : IsEmpty α
    f : α → M
    ⊢ Eq (finprod fun i => f i) 1
  -/
  rw [← finprod_one]
  /-
    M : Type u_2
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : IsEmpty α
    f : α → M
    ⊢ Eq (finprod fun i => f i) (finprod fun x => 1)
  -/
  congr
  /-
    case e_f
    M : Type u_2
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : IsEmpty α
    f : α → M
    ⊢ Eq (fun i => f i) fun x => 1
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem finprod_false (f : False → M) : ∏ᶠ i, f i = 1 :=
  finprod_of_isEmpty _


@[to_additive]
theorem finprod_eq_single (f : α → M) (a : α) (ha : ∀ x, x ≠ a → f x = 1) :
    ∏ᶠ x, f x = f a := by
  have : mulSupport (f ∘ PLift.down) ⊆ ({PLift.up a} : Finset (PLift α)) := by
    intro x
    contrapose
    simpa [PLift.eq_up_iff_down_eq] using ha x.down
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    a : α
    ha : ∀ (x : α), Ne x a → Eq (f x) 1
    this : HasSubset.Subset (Function.mulSupport (Function.comp f PLift.down)) ↑(S …
    ⊢ Eq (finprod fun x => f x) (f a)
  -/
  rw [finprod_eq_prod_plift_of_mulSupport_subset this, Finset.prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_unique [Unique α] (f : α → M) : ∏ᶠ i, f i = f default :=
  finprod_eq_single f default fun _x hx => (hx <| Unique.eq_default _).elim


@[to_additive (attr := simp)]
theorem finprod_true (f : True → M) : ∏ᶠ i, f i = f trivial :=
  @finprod_unique M True _ ⟨⟨trivial⟩, fun _ => rfl⟩ f


@[to_additive]
theorem finprod_eq_dif {p : Prop} [Decidable p] (f : p → M) :
    ∏ᶠ i, f i = if h : p then f h else 1 := by
  /-
    M : Type u_2
    inst✝¹ : CommMonoid M
    p : Prop
    inst✝ : Decidable p
    f : p → M
    ⊢ Eq (finprod fun i => f i) (dite p (fun h => f h) fun h => 1)
  -/
  split_ifs with h
    /-
      case pos
      M : Type u_2
      inst✝¹ : CommMonoid M
      p : Prop
      inst✝ : Decidable p
      f : p → M
      h : p
      ⊢ Eq (finprod fun i => f i) (f h)
    -/
  · haveI : Unique p := ⟨⟨h⟩, fun _ => rfl⟩
    /-
      case pos
      M : Type u_2
      inst✝¹ : CommMonoid M
      p : Prop
      inst✝ : Decidable p
      f : p → M
      h : p
      this : Unique p
      ⊢ Eq (finprod fun i => f i) (f h)
    -/
    exact finprod_unique f
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_2
      inst✝¹ : CommMonoid M
      p : Prop
      inst✝ : Decidable p
      f : p → M
      h : Not p
      ⊢ Eq (finprod fun i => f i) 1
    -/
  · haveI : IsEmpty p := ⟨h⟩
    /-
      case neg
      M : Type u_2
      inst✝¹ : CommMonoid M
      p : Prop
      inst✝ : Decidable p
      f : p → M
      h : Not p
      this : IsEmpty p
      ⊢ Eq (finprod fun i => f i) 1
    -/
    exact finprod_of_isEmpty f
    /-
      🎉 no goals
    -/


@[to_additive]
theorem finprod_eq_if {p : Prop} [Decidable p] {x : M} : ∏ᶠ _ : p, x = if p then x else 1 :=
  finprod_eq_dif fun _ => x


@[to_additive]
theorem finprod_congr {f g : α → M} (h : ∀ x, f x = g x) : finprod f = finprod g :=
  congr_arg _ <| funext h


@[to_additive (attr := congr)]
theorem finprod_congr_Prop {p q : Prop} {f : p → M} {g : q → M} (hpq : p = q)
    (hfg : ∀ h : q, f (hpq.mpr h) = g h) : finprod f = finprod g := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    p q : Prop
    f : p → M
    g : q → M
    hpq : Eq p q
    hfg : ∀ (h : q), Eq (f ⋯) (g h)
    ⊢ Eq (finprod f) (finprod g)
  -/
  subst q
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    p : Prop
    f g : p → M
    hfg : ∀ (h : p), Eq (f ⋯) (g h)
    ⊢ Eq (finprod f) (finprod g)
  -/
  exact finprod_congr hfg
  /-
    🎉 no goals
  -/


/-- To prove a property of a finite product, it suffices to prove that the property is
multiplicative and holds on the factors. -/
@[to_additive
      "To prove a property of a finite sum, it suffices to prove that the property is
      additive and holds on the summands."]
theorem finprod_induction {f : α → M} (p : M → Prop) (hp₀ : p 1)
    (hp₁ : ∀ x y, p x → p y → p (x * y)) (hp₂ : ∀ i, p (f i)) : p (∏ᶠ i, f i) := by
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    p : M → Prop
    hp₀ : p 1
    hp₁ : ∀ (x y : M), p x → p y → p (HMul.hMul x y)
    hp₂ : ∀ (i : α), p (f i)
    ⊢ p (finprod fun i => f i)
  -/
  rw [finprod]
  /-
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    p : M → Prop
    hp₀ : p 1
    hp₁ : ∀ (x y : M), p x → p y → p (HMul.hMul x y)
    hp₂ : ∀ (i : α), p (f i)
    ⊢ p (dite (Function.mulSupport (Function.comp (fun i => f i) PLift.down)).Fini …
  -/
  split_ifs
  /-
    case pos
    M : Type u_2
    α : Sort u_4
    inst✝ : CommMonoid M
    f : α → M
    p : M → Prop
    hp₀ : p 1
    hp₁ : ∀ (x y : M), p x → p y → p (HMul.hMul x y)
    hp₂ : ∀ (i : α), p (f i)
    h✝ : (Function.mulSupport (Function.comp (fun i => f i) PLift.down)).Finite
    ⊢ p (h✝.toFinset.prod fun i => f i.down)
  -/
  exacts [Finset.prod_induction _ _ hp₁ hp₀ fun i _ => hp₂ _, hp₀]
  /-
    🎉 no goals
  -/


theorem finprod_nonneg {R : Type*} [OrderedCommSemiring R] {f : α → R} (hf : ∀ x, 0 ≤ f x) :
    0 ≤ ∏ᶠ x, f x :=
  finprod_induction (fun x => 0 ≤ x) zero_le_one (fun _ _ => mul_nonneg) hf


@[to_additive finsum_nonneg]
theorem one_le_finprod' {M : Type*} [OrderedCommMonoid M] {f : α → M} (hf : ∀ i, 1 ≤ f i) :
    1 ≤ ∏ᶠ i, f i :=
  finprod_induction _ le_rfl (fun _ _ => one_le_mul) hf


@[to_additive]
theorem MonoidHom.map_finprod_plift (f : M →* N) (g : α → M)
    (h : (mulSupport <| g ∘ PLift.down).Finite) : f (∏ᶠ x, g x) = ∏ᶠ x, f (g x) := by
  rw [finprod_eq_prod_plift_of_mulSupport_subset h.coe_toFinset.ge,
    finprod_eq_prod_plift_of_mulSupport_subset, map_prod]
  /-
    M : Type u_2
    N : Type u_3
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom M N
    g : α → M
    h : (Function.mulSupport (Function.comp g PLift.down)).Finite
    ⊢ HasSubset.Subset (Function.mulSupport (Function.comp (fun x => f (g x)) PLif …
  -/
  rw [h.coe_toFinset]
  /-
    M : Type u_2
    N : Type u_3
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom M N
    g : α → M
    h : (Function.mulSupport (Function.comp g PLift.down)).Finite
    ⊢ HasSubset.Subset (Function.mulSupport (Function.comp (fun x => f (g x)) PLif …
  -/
  exact mulSupport_comp_subset f.map_one (g ∘ PLift.down)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem MonoidHom.map_finprod_Prop {p : Prop} (f : M →* N) (g : p → M) :
    f (∏ᶠ x, g x) = ∏ᶠ x, f (g x) :=
  f.map_finprod_plift g (Set.toFinite _)


@[to_additive]
theorem MonoidHom.map_finprod_of_preimage_one (f : M →* N) (hf : ∀ x, f x = 1 → x = 1) (g : α → M) :
    f (∏ᶠ i, g i) = ∏ᶠ i, f (g i) := by
  /-
    M : Type u_2
    N : Type u_3
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom M N
    hf : ∀ (x : M), Eq (f x) 1 → Eq x 1
    g : α → M
    ⊢ Eq (f (finprod fun i => g i)) (finprod fun i => f (g i))
  -/
  by_cases hg : (mulSupport <| g ∘ PLift.down).Finite; · exact f.map_finprod_plift g hg
                                                         /-
                                                           🎉 no goals
                                                         -/
  /-
    case neg
    M : Type u_2
    N : Type u_3
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom M N
    hf : ∀ (x : M), Eq (f x) 1 → Eq x 1
    g : α → M
    hg : Not (Function.mulSupport (Function.comp g PLift.down)).Finite
    ⊢ Eq (f (finprod fun i => g i)) (finprod fun i => f (g i))
  -/
  rw [finprod, dif_neg, f.map_one, finprod, dif_neg]
  /-
    case neg.hnc
    M : Type u_2
    N : Type u_3
    α : Sort u_4
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom M N
    hf : ∀ (x : M), Eq (f x) 1 → Eq x 1
    g : α → M
    hg : Not (Function.mulSupport (Function.comp g PLift.down)).Finite
    ⊢ Not (Function.mulSupport (Function.comp (fun i => f (g i)) PLift.down)).Finite
  -/
  exacts [Infinite.mono (fun x hx => mt (hf (g x.down)) hx) hg, hg]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem MonoidHom.map_finprod_of_injective (g : M →* N) (hg : Injective g) (f : α → M) :
    g (∏ᶠ i, f i) = ∏ᶠ i, g (f i) :=
  g.map_finprod_of_preimage_one (fun _ => (hg.eq_iff' g.map_one).mp) f


@[to_additive]
theorem MulEquiv.map_finprod (g : M ≃* N) (f : α → M) : g (∏ᶠ i, f i) = ∏ᶠ i, g (f i) :=
  g.toMonoidHom.map_finprod_of_injective (EquivLike.injective g) f


/-- The `NoZeroSMulDivisors` makes sure that the result holds even when the support of `f` is
infinite. For a more usual version assuming `(support f).Finite` instead, see `finsum_smul'`. -/
theorem finsum_smul {R M : Type*} [Ring R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M]
    (f : ι → R) (x : M) : (∑ᶠ i, f i) • x = ∑ᶠ i, f i • x := by
  /-
    ι : Sort u_6
    R : Type u_7
    M : Type u_8
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    f : ι → R
    x : M
    ⊢ Eq (HSMul.hSMul (finsum fun i => f i) x) (finsum fun i => HSMul.hSMul (f i) x)
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case inl
      ι : Sort u_6
      R : Type u_7
      M : Type u_8
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      f : ι → R
      ⊢ Eq (HSMul.hSMul (finsum fun i => f i) 0) (finsum fun i => HSMul.hSMul (f i) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_6
      R : Type u_7
      M : Type u_8
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      f : ι → R
      x : M
      hx : Ne x 0
      ⊢ Eq (HSMul.hSMul (finsum fun i => f i) x) (finsum fun i => HSMul.hSMul (f i) x)
    -/
  · exact ((smulAddHom R M).flip x).map_finsum_of_injective (smul_left_injective R hx) _
    /-
      🎉 no goals
    -/


/-- The `NoZeroSMulDivisors` makes sure that the result holds even when the support of `f` is
infinite. For a more usual version assuming `(support f).Finite` instead, see `smul_finsum'`. -/
theorem smul_finsum {R M : Type*} [Ring R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M]
    (c : R) (f : ι → M) : (c • ∑ᶠ i, f i) = ∑ᶠ i, c • f i := by
  /-
    ι : Sort u_6
    R : Type u_7
    M : Type u_8
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    c : R
    f : ι → M
    ⊢ Eq (HSMul.hSMul c (finsum fun i => f i)) (finsum fun i => HSMul.hSMul c (f i))
  -/
  rcases eq_or_ne c 0 with (rfl | hc)
    /-
      case inl
      ι : Sort u_6
      R : Type u_7
      M : Type u_8
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      f : ι → M
      ⊢ Eq (HSMul.hSMul 0 (finsum fun i => f i)) (finsum fun i => HSMul.hSMul 0 (f i))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_6
      R : Type u_7
      M : Type u_8
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      c : R
      f : ι → M
      hc : Ne c 0
      ⊢ Eq (HSMul.hSMul c (finsum fun i => f i)) (finsum fun i => HSMul.hSMul c (f i))
    -/
  · exact (smulAddHom R M c).map_finsum_of_injective (smul_right_injective M hc) _
    /-
      🎉 no goals
    -/


@[to_additive]
theorem finprod_inv_distrib [DivisionCommMonoid G] (f : α → G) : (∏ᶠ x, (f x)⁻¹) = (∏ᶠ x, f x)⁻¹ :=
  ((MulEquiv.inv G).map_finprod f).symm


@[to_additive]
theorem finprod_eq_mulIndicator_apply (s : Set α) (f : α → M) (a : α) :
    ∏ᶠ _ : a ∈ s, f a = mulIndicator s f a := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    s : Set α
    f : α → M
    a : α
    ⊢ Eq (finprod fun x => f a) (s.mulIndicator f a)
  -/
  classical convert finprod_eq_if (M := M) (p := a ∈ s) (x := f a)
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem finprod_mem_mulSupport (f : α → M) (a : α) : ∏ᶠ _ : f a ≠ 1, f a = f a := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    a : α
    ⊢ Eq (finprod fun x => f a) (f a)
  -/
  rw [← mem_mulSupport, finprod_eq_mulIndicator_apply, mulIndicator_mulSupport]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_def (s : Set α) (f : α → M) : ∏ᶠ a ∈ s, f a = ∏ᶠ a, mulIndicator s f a :=
  finprod_congr <| finprod_eq_mulIndicator_apply s f


@[to_additive]
theorem finprod_eq_prod_of_mulSupport_subset (f : α → M) {s : Finset α} (h : mulSupport f ⊆ s) :
    ∏ᶠ i, f i = ∏ i ∈ s, f i := by
  have A : mulSupport (f ∘ PLift.down) = Equiv.plift.symm '' mulSupport f := by
    rw [mulSupport_comp_eq_preimage]
    exact (Equiv.plift.symm.image_eq_preimage _).symm
  have : mulSupport (f ∘ PLift.down) ⊆ s.map Equiv.plift.symm.toEmbedding := by
    rw [A, Finset.coe_map]
    exact image_subset _ h
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Finset α
    h : HasSubset.Subset (Function.mulSupport f) ↑s
    A : Eq (Function.mulSupport (Function.comp f PLift.down)) (Set.image (⇑Equiv.p …
    this : HasSubset.Subset (Function.mulSupport (Function.comp f PLift.down)) ↑(F …
    ⊢ Eq (finprod fun i => f i) (s.prod fun i => f i)
  -/
  rw [finprod_eq_prod_plift_of_mulSupport_subset this]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Finset α
    h : HasSubset.Subset (Function.mulSupport f) ↑s
    A : Eq (Function.mulSupport (Function.comp f PLift.down)) (Set.image (⇑Equiv.p …
    this : HasSubset.Subset (Function.mulSupport (Function.comp f PLift.down)) ↑(F …
    ⊢ Eq ((Finset.map Equiv.plift.symm.toEmbedding s).prod fun i => f i.down) (s.p …
  -/
  simp only [Finset.prod_map, Equiv.coe_toEmbedding]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Finset α
    h : HasSubset.Subset (Function.mulSupport f) ↑s
    A : Eq (Function.mulSupport (Function.comp f PLift.down)) (Set.image (⇑Equiv.p …
    this : HasSubset.Subset (Function.mulSupport (Function.comp f PLift.down)) ↑(F …
    ⊢ Eq (s.prod fun x => f (Equiv.plift.symm x).down) (s.prod fun i => f i)
  -/
  congr
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_eq_prod_of_mulSupport_toFinset_subset (f : α → M) (hf : (mulSupport f).Finite)
    {s : Finset α} (h : hf.toFinset ⊆ s) : ∏ᶠ i, f i = ∏ i ∈ s, f i :=
  finprod_eq_prod_of_mulSupport_subset _ fun _ hx => h <| hf.mem_toFinset.2 hx


@[to_additive]
theorem finprod_eq_finset_prod_of_mulSupport_subset (f : α → M) {s : Finset α}
    (h : mulSupport f ⊆ (s : Set α)) : ∏ᶠ i, f i = ∏ i ∈ s, f i :=
  haveI h' : (s.finite_toSet.subset h).toFinset ⊆ s := by
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      f : α → M
      s : Finset α
      h : HasSubset.Subset (Function.mulSupport f) ↑s
      ⊢ HasSubset.Subset ⋯.toFinset s
    -/
    simpa [← Finset.coe_subset, Set.coe_toFinset]
    /-
      🎉 no goals
    -/
  finprod_eq_prod_of_mulSupport_toFinset_subset _ _ h'


@[to_additive]
theorem finprod_def (f : α → M) [Decidable (mulSupport f).Finite] :
    ∏ᶠ i : α, f i = if h : (mulSupport f).Finite then ∏ i ∈ h.toFinset, f i else 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : Decidable (Function.mulSupport f).Finite
    ⊢ Eq (finprod fun i => f i) (dite (Function.mulSupport f).Finite (fun h => h.t …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      M : Type u_5
      inst✝¹ : CommMonoid M
      f : α → M
      inst✝ : Decidable (Function.mulSupport f).Finite
      h : (Function.mulSupport f).Finite
      ⊢ Eq (finprod fun i => f i) (h.toFinset.prod fun i => f i)
    -/
  · exact finprod_eq_prod_of_mulSupport_toFinset_subset _ h (Finset.Subset.refl _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      M : Type u_5
      inst✝¹ : CommMonoid M
      f : α → M
      inst✝ : Decidable (Function.mulSupport f).Finite
      h : Not (Function.mulSupport f).Finite
      ⊢ Eq (finprod fun i => f i) 1
    -/
  · rw [finprod, dif_neg]
    /-
      case neg.hnc
      α : Type u_1
      M : Type u_5
      inst✝¹ : CommMonoid M
      f : α → M
      inst✝ : Decidable (Function.mulSupport f).Finite
      h : Not (Function.mulSupport f).Finite
      ⊢ Not (Function.mulSupport (Function.comp (fun i => f i) PLift.down)).Finite
    -/
    rw [mulSupport_comp_eq_preimage]
    /-
      case neg.hnc
      α : Type u_1
      M : Type u_5
      inst✝¹ : CommMonoid M
      f : α → M
      inst✝ : Decidable (Function.mulSupport f).Finite
      h : Not (Function.mulSupport f).Finite
      ⊢ Not (Set.preimage PLift.down (Function.mulSupport fun i => f i)).Finite
    -/
    exact mt (fun hf => hf.of_preimage Equiv.plift.surjective) h
    /-
      🎉 no goals
    -/


@[to_additive]
theorem finprod_of_infinite_mulSupport {f : α → M} (hf : (mulSupport f).Infinite) :
                        /-
                          α : Type u_1
                          M : Type u_5
                          inst✝ : CommMonoid M
                          f : α → M
                          hf : (Function.mulSupport f).Infinite
                          ⊢ Eq (finprod fun i => f i) 1
                        -/
    ∏ᶠ i, f i = 1 := by classical rw [finprod_def, dif_neg hf]
                        /-
                          🎉 no goals
                        -/


@[to_additive]
theorem finprod_eq_prod (f : α → M) (hf : (mulSupport f).Finite) :
                                                 /-
                                                   α : Type u_1
                                                   M : Type u_5
                                                   inst✝ : CommMonoid M
                                                   f : α → M
                                                   hf : (Function.mulSupport f).Finite
                                                   ⊢ Eq (finprod fun i => f i) (hf.toFinset.prod fun i => f i)
                                                 -/
    ∏ᶠ i : α, f i = ∏ i ∈ hf.toFinset, f i := by classical rw [finprod_def, dif_pos hf]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive]
theorem finprod_eq_prod_of_fintype [Fintype α] (f : α → M) : ∏ᶠ i : α, f i = ∏ i, f i :=
  finprod_eq_prod_of_mulSupport_toFinset_subset _ (Set.toFinite _) <| Finset.subset_univ _


@[to_additive]
theorem finprod_cond_eq_prod_of_cond_iff (f : α → M) {p : α → Prop} {t : Finset α}
    (h : ∀ {x}, f x ≠ 1 → (p x ↔ x ∈ t)) : (∏ᶠ (i) (_ : p i), f i) = ∏ i ∈ t, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    p : α → Prop
    t : Finset α
    h : ∀ {x : α}, Ne (f x) 1 → Iff (p x) (Membership.mem t x)
    ⊢ Eq (finprod fun i => finprod fun x => f i) (t.prod fun i => f i)
  -/
  set s := { x | p x }
  have : mulSupport (s.mulIndicator f) ⊆ t := by
    rw [Set.mulSupport_mulIndicator]
    intro x hx
    exact (h hx.2).1 hx.1
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    p : α → Prop
    t : Finset α
    h : ∀ {x : α}, Ne (f x) 1 → Iff (p x) (Membership.mem t x)
    s : Set α := setOf fun x => p x
    this : HasSubset.Subset (Function.mulSupport (s.mulIndicator f)) ↑t
    ⊢ Eq (finprod fun i => finprod fun x => f i) (t.prod fun i => f i)
  -/
  erw [finprod_mem_def, finprod_eq_prod_of_mulSupport_subset _ this]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    p : α → Prop
    t : Finset α
    h : ∀ {x : α}, Ne (f x) 1 → Iff (p x) (Membership.mem t x)
    s : Set α := setOf fun x => p x
    this : HasSubset.Subset (Function.mulSupport (s.mulIndicator f)) ↑t
    ⊢ Eq (t.prod fun i => s.mulIndicator f i) (t.prod fun i => f i)
  -/
  refine Finset.prod_congr rfl fun x hx => mulIndicator_apply_eq_self.2 fun hxs => ?_
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    p : α → Prop
    t : Finset α
    h : ∀ {x : α}, Ne (f x) 1 → Iff (p x) (Membership.mem t x)
    s : Set α := setOf fun x => p x
    this : HasSubset.Subset (Function.mulSupport (s.mulIndicator f)) ↑t
    x : α
    hx : Membership.mem t x
    hxs : Not (Membership.mem s x)
    ⊢ Eq (f x) 1
  -/
  contrapose! hxs
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    p : α → Prop
    t : Finset α
    h : ∀ {x : α}, Ne (f x) 1 → Iff (p x) (Membership.mem t x)
    s : Set α := setOf fun x => p x
    this : HasSubset.Subset (Function.mulSupport (s.mulIndicator f)) ↑t
    x : α
    hx : Membership.mem t x
    hxs : Ne (f x) 1
    ⊢ Membership.mem s x
  -/
  exact (h hxs).2 hx
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_cond_ne (f : α → M) (a : α) [DecidableEq α] (hf : (mulSupport f).Finite) :
    (∏ᶠ (i) (_ : i ≠ a), f i) = ∏ i ∈ hf.toFinset.erase a, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    a : α
    inst✝ : DecidableEq α
    hf : (Function.mulSupport f).Finite
    ⊢ Eq (finprod fun i => finprod fun x => f i) ((hf.toFinset.erase a).prod fun i …
  -/
  apply finprod_cond_eq_prod_of_cond_iff
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    a : α
    inst✝ : DecidableEq α
    hf : (Function.mulSupport f).Finite
    ⊢ ∀ {x : α}, Ne (f x) 1 → Iff (Ne x a) (Membership.mem (hf.toFinset.erase a) x)
  -/
  intro x hx
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    a : α
    inst✝ : DecidableEq α
    hf : (Function.mulSupport f).Finite
    x : α
    hx : Ne (f x) 1
    ⊢ Iff (Ne x a) (Membership.mem (hf.toFinset.erase a) x)
  -/
  rw [Finset.mem_erase, Finite.mem_toFinset, mem_mulSupport]
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    a : α
    inst✝ : DecidableEq α
    hf : (Function.mulSupport f).Finite
    x : α
    hx : Ne (f x) 1
    ⊢ Iff (Ne x a) (And (Ne x a) (Ne (f x) 1))
  -/
  exact ⟨fun h => And.intro h hx, fun h => h.1⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_eq_prod_of_inter_mulSupport_eq (f : α → M) {s : Set α} {t : Finset α}
    (h : s ∩ mulSupport f = t.toSet ∩ mulSupport f) : ∏ᶠ i ∈ s, f i = ∏ i ∈ t, f i :=
  finprod_cond_eq_prod_of_cond_iff _ <| by
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      f : α → M
      s : Set α
      t : Finset α
      h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
      ⊢ ∀ {x : α}, Ne (f x) 1 → Iff (Membership.mem s x) (Membership.mem t x)
    -/
    intro x hxf
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      f : α → M
      s : Set α
      t : Finset α
      h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
      x : α
      hxf : Ne (f x) 1
      ⊢ Iff (Membership.mem s x) (Membership.mem t x)
    -/
    rw [← mem_mulSupport] at hxf
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      f : α → M
      s : Set α
      t : Finset α
      h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
      x : α
      hxf : Membership.mem (Function.mulSupport f) x
      ⊢ Iff (Membership.mem s x) (Membership.mem t x)
    -/
    refine ⟨fun hx => ?_, fun hx => ?_⟩
      /-
        case refine_1
        α : Type u_1
        M : Type u_5
        inst✝ : CommMonoid M
        f : α → M
        s : Set α
        t : Finset α
        h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
        x : α
        hxf : Membership.mem (Function.mulSupport f) x
        hx : Membership.mem s x
        ⊢ Membership.mem t x
      -/
    · refine ((mem_inter_iff x t (mulSupport f)).mp ?_).1
      /-
        case refine_1
        α : Type u_1
        M : Type u_5
        inst✝ : CommMonoid M
        f : α → M
        s : Set α
        t : Finset α
        h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
        x : α
        hxf : Membership.mem (Function.mulSupport f) x
        hx : Membership.mem s x
        ⊢ Membership.mem (Inter.inter (↑t) (Function.mulSupport f)) x
      -/
      rw [← Set.ext_iff.mp h x, mem_inter_iff]
      /-
        case refine_1
        α : Type u_1
        M : Type u_5
        inst✝ : CommMonoid M
        f : α → M
        s : Set α
        t : Finset α
        h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
        x : α
        hxf : Membership.mem (Function.mulSupport f) x
        hx : Membership.mem s x
        ⊢ And (Membership.mem s x) (Membership.mem (Function.mulSupport f) x)
      -/
      exact ⟨hx, hxf⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        M : Type u_5
        inst✝ : CommMonoid M
        f : α → M
        s : Set α
        t : Finset α
        h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
        x : α
        hxf : Membership.mem (Function.mulSupport f) x
        hx : Membership.mem t x
        ⊢ Membership.mem s x
      -/
    · refine ((mem_inter_iff x s (mulSupport f)).mp ?_).1
      /-
        case refine_2
        α : Type u_1
        M : Type u_5
        inst✝ : CommMonoid M
        f : α → M
        s : Set α
        t : Finset α
        h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
        x : α
        hxf : Membership.mem (Function.mulSupport f) x
        hx : Membership.mem t x
        ⊢ Membership.mem (Inter.inter s (Function.mulSupport f)) x
      -/
      rw [Set.ext_iff.mp h x, mem_inter_iff]
      /-
        case refine_2
        α : Type u_1
        M : Type u_5
        inst✝ : CommMonoid M
        f : α → M
        s : Set α
        t : Finset α
        h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑t) (Function.mul …
        x : α
        hxf : Membership.mem (Function.mulSupport f) x
        hx : Membership.mem t x
        ⊢ And (Membership.mem (↑t) x) (Membership.mem (Function.mulSupport f) x)
      -/
      exact ⟨hx, hxf⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem finprod_mem_eq_prod_of_subset (f : α → M) {s : Set α} {t : Finset α}
    (h₁ : s ∩ mulSupport f ⊆ t) (h₂ : ↑t ⊆ s) : ∏ᶠ i ∈ s, f i = ∏ i ∈ t, f i :=
  finprod_cond_eq_prod_of_cond_iff _ fun hx => ⟨fun h => h₁ ⟨h, hx⟩, fun h => h₂ h⟩


@[to_additive]
theorem finprod_mem_eq_prod (f : α → M) {s : Set α} (hf : (s ∩ mulSupport f).Finite) :
    ∏ᶠ i ∈ s, f i = ∏ i ∈ hf.toFinset, f i :=
                                                     /-
                                                       α : Type u_1
                                                       M : Type u_5
                                                       inst✝ : CommMonoid M
                                                       f : α → M
                                                       s : Set α
                                                       hf : (Inter.inter s (Function.mulSupport f)).Finite
                                                       ⊢ Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑hf.toFinset) (Func …
                                                     -/
  finprod_mem_eq_prod_of_inter_mulSupport_eq _ <| by simp [inter_assoc]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
theorem finprod_mem_eq_prod_filter (f : α → M) (s : Set α) [DecidablePred (· ∈ s)]
    (hf : (mulSupport f).Finite) :
    ∏ᶠ i ∈ s, f i = ∏ i ∈ Finset.filter (· ∈ s) hf.toFinset, f i :=
  finprod_mem_eq_prod_of_inter_mulSupport_eq _ <| by
    /-
      α : Type u_1
      M : Type u_5
      inst✝¹ : CommMonoid M
      f : α → M
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hf : (Function.mulSupport f).Finite
      ⊢ Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑(Finset.filter (fu …
    -/
    ext x
    /-
      case h
      α : Type u_1
      M : Type u_5
      inst✝¹ : CommMonoid M
      f : α → M
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hf : (Function.mulSupport f).Finite
      x : α
      ⊢ Iff (Membership.mem (Inter.inter s (Function.mulSupport f)) x) (Membership.m …
    -/
    simp [and_comm]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem finprod_mem_eq_toFinset_prod (f : α → M) (s : Set α) [Fintype s] :
    ∏ᶠ i ∈ s, f i = ∏ i ∈ s.toFinset, f i :=
                                                     /-
                                                       α : Type u_1
                                                       M : Type u_5
                                                       inst✝¹ : CommMonoid M
                                                       f : α → M
                                                       s : Set α
                                                       inst✝ : Fintype ↑s
                                                       ⊢ Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑s.toFinset) (Funct …
                                                     -/
  finprod_mem_eq_prod_of_inter_mulSupport_eq _ <| by simp_rw [coe_toFinset s]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
theorem finprod_mem_eq_finite_toFinset_prod (f : α → M) {s : Set α} (hs : s.Finite) :
    ∏ᶠ i ∈ s, f i = ∏ i ∈ hs.toFinset, f i :=
                                                     /-
                                                       α : Type u_1
                                                       M : Type u_5
                                                       inst✝ : CommMonoid M
                                                       f : α → M
                                                       s : Set α
                                                       hs : s.Finite
                                                       ⊢ Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter (↑hs.toFinset) (Func …
                                                     -/
  finprod_mem_eq_prod_of_inter_mulSupport_eq _ <| by rw [hs.coe_toFinset]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
theorem finprod_mem_finset_eq_prod (f : α → M) (s : Finset α) : ∏ᶠ i ∈ s, f i = ∏ i ∈ s, f i :=
  finprod_mem_eq_prod_of_inter_mulSupport_eq _ rfl


@[to_additive]
theorem finprod_mem_coe_finset (f : α → M) (s : Finset α) :
    (∏ᶠ i ∈ (s : Set α), f i) = ∏ i ∈ s, f i :=
  finprod_mem_eq_prod_of_inter_mulSupport_eq _ rfl


@[to_additive]
theorem finprod_mem_eq_one_of_infinite {f : α → M} {s : Set α} (hs : (s ∩ mulSupport f).Infinite) :
    ∏ᶠ i ∈ s, f i = 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    hs : (Inter.inter s (Function.mulSupport f)).Infinite
    ⊢ Eq (finprod fun i => finprod fun h => f i) 1
  -/
  rw [finprod_mem_def]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    hs : (Inter.inter s (Function.mulSupport f)).Infinite
    ⊢ Eq (finprod fun a => s.mulIndicator f a) 1
  -/
  apply finprod_of_infinite_mulSupport
  /-
    case hf
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    hs : (Inter.inter s (Function.mulSupport f)).Infinite
    ⊢ (Function.mulSupport (s.mulIndicator f)).Infinite
  -/
  rwa [← mulSupport_mulIndicator] at hs
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_eq_one_of_forall_eq_one {f : α → M} {s : Set α} (h : ∀ x ∈ s, f x = 1) :
                            /-
                              α : Type u_1
                              M : Type u_5
                              inst✝ : CommMonoid M
                              f : α → M
                              s : Set α
                              h : ∀ (x : α), Membership.mem s x → Eq (f x) 1
                              ⊢ Eq (finprod fun i => finprod fun h => f i) 1
                            -/
    ∏ᶠ i ∈ s, f i = 1 := by simp +contextual [h]
                            /-
                              🎉 no goals
                            -/


@[to_additive]
theorem finprod_mem_inter_mulSupport (f : α → M) (s : Set α) :
    ∏ᶠ i ∈ s ∩ mulSupport f, f i = ∏ᶠ i ∈ s, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun i => finprod fun h = …
  -/
  rw [finprod_mem_def, finprod_mem_def, mulIndicator_inter_mulSupport]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_inter_mulSupport_eq (f : α → M) (s t : Set α)
    (h : s ∩ mulSupport f = t ∩ mulSupport f) : ∏ᶠ i ∈ s, f i = ∏ᶠ i ∈ t, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    h : Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter t (Function.mulSup …
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun i => finprod fun h = …
  -/
  rw [← finprod_mem_inter_mulSupport, h, finprod_mem_inter_mulSupport]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_inter_mulSupport_eq' (f : α → M) (s t : Set α)
    (h : ∀ x ∈ mulSupport f, x ∈ s ↔ x ∈ t) : ∏ᶠ i ∈ s, f i = ∏ᶠ i ∈ t, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    h : ∀ (x : α), Membership.mem (Function.mulSupport f) x → Iff (Membership.mem  …
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun i => finprod fun h = …
  -/
  apply finprod_mem_inter_mulSupport_eq
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    h : ∀ (x : α), Membership.mem (Function.mulSupport f) x → Iff (Membership.mem  …
    ⊢ Eq (Inter.inter s (Function.mulSupport f)) (Inter.inter t (Function.mulSuppo …
  -/
  ext x
  /-
    case h.h
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    h : ∀ (x : α), Membership.mem (Function.mulSupport f) x → Iff (Membership.mem  …
    x : α
    ⊢ Iff (Membership.mem (Inter.inter s (Function.mulSupport f)) x) (Membership.m …
  -/
  exact and_congr_left (h x)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_univ (f : α → M) : ∏ᶠ i ∈ @Set.univ α, f i = ∏ᶠ i : α, f i :=
  finprod_congr fun _ => finprod_true _


@[to_additive]
theorem finprod_mem_congr (h₀ : s = t) (h₁ : ∀ x ∈ t, f x = g x) :
    ∏ᶠ i ∈ s, f i = ∏ᶠ i ∈ t, g i :=
  h₀.symm ▸ finprod_congr fun i => finprod_congr_Prop rfl (h₁ i)


@[to_additive]
theorem finprod_eq_one_of_forall_eq_one {f : α → M} (h : ∀ x, f x = 1) : ∏ᶠ i, f i = 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    h : ∀ (x : α), Eq (f x) 1
    ⊢ Eq (finprod fun i => f i) 1
  -/
  simp +contextual [h]
  /-
    🎉 no goals
  -/


@[to_additive finsum_pos']
theorem one_lt_finprod' {M : Type*} [OrderedCancelCommMonoid M] {f : ι → M}
    (h : ∀ i, 1 ≤ f i) (h' : ∃ i, 1 < f i) (hf : (mulSupport f).Finite) : 1 < ∏ᶠ i, f i := by
  /-
    ι : Type u_3
    M : Type u_7
    inst✝ : OrderedCancelCommMonoid M
    f : ι → M
    h : ∀ (i : ι), LE.le 1 (f i)
    h' : Exists fun i => LT.lt 1 (f i)
    hf : (Function.mulSupport f).Finite
    ⊢ LT.lt 1 (finprod fun i => f i)
  -/
  rcases h' with ⟨i, hi⟩
  /-
    case intro
    ι : Type u_3
    M : Type u_7
    inst✝ : OrderedCancelCommMonoid M
    f : ι → M
    h : ∀ (i : ι), LE.le 1 (f i)
    hf : (Function.mulSupport f).Finite
    i : ι
    hi : LT.lt 1 (f i)
    ⊢ LT.lt 1 (finprod fun i => f i)
  -/
  rw [finprod_eq_prod _ hf]
  /-
    case intro
    ι : Type u_3
    M : Type u_7
    inst✝ : OrderedCancelCommMonoid M
    f : ι → M
    h : ∀ (i : ι), LE.le 1 (f i)
    hf : (Function.mulSupport f).Finite
    i : ι
    hi : LT.lt 1 (f i)
    ⊢ LT.lt 1 (hf.toFinset.prod fun i => f i)
  -/
  refine Finset.one_lt_prod' (fun i _ ↦ h i) ⟨i, ?_, hi⟩
  /-
    case intro
    ι : Type u_3
    M : Type u_7
    inst✝ : OrderedCancelCommMonoid M
    f : ι → M
    h : ∀ (i : ι), LE.le 1 (f i)
    hf : (Function.mulSupport f).Finite
    i : ι
    hi : LT.lt 1 (f i)
    ⊢ Membership.mem hf.toFinset i
  -/
  simpa only [Finite.mem_toFinset, mem_mulSupport] using ne_of_gt hi
  /-
    🎉 no goals
  -/


/-- If the multiplicative supports of `f` and `g` are finite, then the product of `f i * g i` equals
the product of `f i` multiplied by the product of `g i`. -/
@[to_additive
      "If the additive supports of `f` and `g` are finite, then the sum of `f i + g i`
      equals the sum of `f i` plus the sum of `g i`."]
theorem finprod_mul_distrib (hf : (mulSupport f).Finite) (hg : (mulSupport g).Finite) :
    ∏ᶠ i, f i * g i = (∏ᶠ i, f i) * ∏ᶠ i, g i := by
  classical
    rw [finprod_eq_prod_of_mulSupport_toFinset_subset f hf Finset.subset_union_left,
      finprod_eq_prod_of_mulSupport_toFinset_subset g hg Finset.subset_union_right, ←
      Finset.prod_mul_distrib]
    refine finprod_eq_prod_of_mulSupport_subset _ ?_
    simp only [Finset.coe_union, Finite.coe_toFinset, mulSupport_subset_iff,
      mem_union, mem_mulSupport]
    intro x
    contrapose!
    rintro ⟨hf, hg⟩
    simp [hf, hg]


/-- If the multiplicative supports of `f` and `g` are finite, then the product of `f i / g i`
equals the product of `f i` divided by the product of `g i`. -/
@[to_additive
      "If the additive supports of `f` and `g` are finite, then the sum of `f i - g i`
      equals the sum of `f i` minus the sum of `g i`."]
theorem finprod_div_distrib [DivisionCommMonoid G] {f g : α → G} (hf : (mulSupport f).Finite)
    (hg : (mulSupport g).Finite) : ∏ᶠ i, f i / g i = (∏ᶠ i, f i) / ∏ᶠ i, g i := by
  simp only [div_eq_mul_inv, finprod_mul_distrib hf ((mulSupport_inv g).symm.rec hg),
    finprod_inv_distrib]


/-- A more general version of `finprod_mem_mul_distrib` that only requires `s ∩ mulSupport f` and
`s ∩ mulSupport g` rather than `s` to be finite. -/
@[to_additive
      "A more general version of `finsum_mem_add_distrib` that only requires `s ∩ support f`
      and `s ∩ support g` rather than `s` to be finite."]
theorem finprod_mem_mul_distrib' (hf : (s ∩ mulSupport f).Finite) (hg : (s ∩ mulSupport g).Finite) :
    ∏ᶠ i ∈ s, f i * g i = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ s, g i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f g : α → M
    s : Set α
    hf : (Inter.inter s (Function.mulSupport f)).Finite
    hg : (Inter.inter s (Function.mulSupport g)).Finite
    ⊢ Eq (finprod fun i => finprod fun h => HMul.hMul (f i) (g i)) (HMul.hMul (fin …
  -/
  rw [← mulSupport_mulIndicator] at hf hg
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f g : α → M
    s : Set α
    hf : (Function.mulSupport (s.mulIndicator f)).Finite
    hg : (Function.mulSupport (s.mulIndicator g)).Finite
    ⊢ Eq (finprod fun i => finprod fun h => HMul.hMul (f i) (g i)) (HMul.hMul (fin …
  -/
  simp only [finprod_mem_def, mulIndicator_mul, finprod_mul_distrib hf hg]
  /-
    🎉 no goals
  -/


/-- The product of the constant function `1` over any set equals `1`. -/
@[to_additive "The sum of the constant function `0` over any set equals `0`."]
                                                                    /-
                                                                      α : Type u_1
                                                                      M : Type u_5
                                                                      inst✝ : CommMonoid M
                                                                      s : Set α
                                                                      ⊢ Eq (finprod fun i => finprod fun h => 1) 1
                                                                    -/
theorem finprod_mem_one (s : Set α) : (∏ᶠ i ∈ s, (1 : M)) = 1 := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- If a function `f` equals `1` on a set `s`, then the product of `f i` over `i ∈ s` equals `1`. -/
@[to_additive
      "If a function `f` equals `0` on a set `s`, then the product of `f i` over `i ∈ s`
      equals `0`."]
theorem finprod_mem_of_eqOn_one (hf : s.EqOn f 1) : ∏ᶠ i ∈ s, f i = 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    hf : Set.EqOn f 1 s
    ⊢ Eq (finprod fun i => finprod fun h => f i) 1
  -/
  rw [← finprod_mem_one s]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    hf : Set.EqOn f 1 s
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun i => finprod fun h = …
  -/
  exact finprod_mem_congr rfl hf
  /-
    🎉 no goals
  -/


/-- If the product of `f i` over `i ∈ s` is not equal to `1`, then there is some `x ∈ s` such that
`f x ≠ 1`. -/
@[to_additive
      "If the product of `f i` over `i ∈ s` is not equal to `0`, then there is some `x ∈ s`
      such that `f x ≠ 0`."]
theorem exists_ne_one_of_finprod_mem_ne_one (h : ∏ᶠ i ∈ s, f i ≠ 1) : ∃ x ∈ s, f x ≠ 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    h : Ne (finprod fun i => finprod fun h => f i) 1
    ⊢ Exists fun x => And (Membership.mem s x) (Ne (f x) 1)
  -/
  by_contra! h'
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    h : Ne (finprod fun i => finprod fun h => f i) 1
    h' : ∀ (x : α), Membership.mem s x → Eq (f x) 1
    ⊢ False
  -/
  exact h (finprod_mem_of_eqOn_one h')
  /-
    🎉 no goals
  -/


/-- Given a finite set `s`, the product of `f i * g i` over `i ∈ s` equals the product of `f i`
over `i ∈ s` times the product of `g i` over `i ∈ s`. -/
@[to_additive
      "Given a finite set `s`, the sum of `f i + g i` over `i ∈ s` equals the sum of `f i`
      over `i ∈ s` plus the sum of `g i` over `i ∈ s`."]
theorem finprod_mem_mul_distrib (hs : s.Finite) :
    ∏ᶠ i ∈ s, f i * g i = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ s, g i :=
  finprod_mem_mul_distrib' (hs.inter_of_left _) (hs.inter_of_left _)


@[to_additive]
theorem MonoidHom.map_finprod {f : α → M} (g : M →* N) (hf : (mulSupport f).Finite) :
    g (∏ᶠ i, f i) = ∏ᶠ i, g (f i) :=
  g.map_finprod_plift f <| hf.preimage Equiv.plift.injective.injOn


@[to_additive]
theorem finprod_pow (hf : (mulSupport f).Finite) (n : ℕ) : (∏ᶠ i, f i) ^ n = ∏ᶠ i, f i ^ n :=
  (powMonoidHom n).map_finprod hf


/-- See also `finsum_smul` for a version that works even when the support of `f` is not finite,
but with slightly stronger typeclass requirements. -/
theorem finsum_smul' {R M : Type*} [Semiring R] [AddCommMonoid M] [Module R M] {f : ι → R}
    (hf : (support f).Finite) (x : M) : (∑ᶠ i, f i) • x = ∑ᶠ i, f i • x :=
  ((smulAddHom R M).flip x).map_finsum hf


/-- See also `smul_finsum` for a version that works even when the support of `f` is not finite,
but with slightly stronger typeclass requirements. -/
theorem smul_finsum' {R M : Type*} [Semiring R] [AddCommMonoid M] [Module R M] (c : R) {f : ι → M}
    (hf : (support f).Finite) : (c • ∑ᶠ i, f i) = ∑ᶠ i, c • f i :=
  (smulAddHom R M c).map_finsum hf


/-- A more general version of `MonoidHom.map_finprod_mem` that requires `s ∩ mulSupport f` rather
than `s` to be finite. -/
@[to_additive
      "A more general version of `AddMonoidHom.map_finsum_mem` that requires
      `s ∩ support f` rather than `s` to be finite."]
theorem MonoidHom.map_finprod_mem' {f : α → M} (g : M →* N) (h₀ : (s ∩ mulSupport f).Finite) :
    g (∏ᶠ j ∈ s, f j) = ∏ᶠ i ∈ s, g (f i) := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_6
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    s : Set α
    f : α → M
    g : MonoidHom M N
    h₀ : (Inter.inter s (Function.mulSupport f)).Finite
    ⊢ Eq (g (finprod fun j => finprod fun h => f j)) (finprod fun i => finprod fun …
  -/
  rw [g.map_finprod]
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_6
      inst✝¹ : CommMonoid M
      inst✝ : CommMonoid N
      s : Set α
      f : α → M
      g : MonoidHom M N
      h₀ : (Inter.inter s (Function.mulSupport f)).Finite
      ⊢ Eq (finprod fun i => g (finprod fun h => f i)) (finprod fun i => finprod fun …
    -/
  · simp only [g.map_finprod_Prop]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_6
      inst✝¹ : CommMonoid M
      inst✝ : CommMonoid N
      s : Set α
      f : α → M
      g : MonoidHom M N
      h₀ : (Inter.inter s (Function.mulSupport f)).Finite
      ⊢ (Function.mulSupport fun j => finprod fun h => f j).Finite
    -/
  · simpa only [finprod_eq_mulIndicator_apply, mulSupport_mulIndicator]
    /-
      🎉 no goals
    -/


/-- Given a monoid homomorphism `g : M →* N` and a function `f : α → M`, the value of `g` at the
product of `f i` over `i ∈ s` equals the product of `g (f i)` over `s`. -/
@[to_additive
      "Given an additive monoid homomorphism `g : M →* N` and a function `f : α → M`, the
      value of `g` at the sum of `f i` over `i ∈ s` equals the sum of `g (f i)` over `s`."]
theorem MonoidHom.map_finprod_mem (f : α → M) (g : M →* N) (hs : s.Finite) :
    g (∏ᶠ j ∈ s, f j) = ∏ᶠ i ∈ s, g (f i) :=
  g.map_finprod_mem' (hs.inter_of_left _)


@[to_additive]
theorem MulEquiv.map_finprod_mem (g : M ≃* N) (f : α → M) {s : Set α} (hs : s.Finite) :
    g (∏ᶠ i ∈ s, f i) = ∏ᶠ i ∈ s, g (f i) :=
  g.toMonoidHom.map_finprod_mem f hs


@[to_additive]
theorem finprod_mem_inv_distrib [DivisionCommMonoid G] (f : α → G) (hs : s.Finite) :
    (∏ᶠ x ∈ s, (f x)⁻¹) = (∏ᶠ x ∈ s, f x)⁻¹ :=
  ((MulEquiv.inv G).map_finprod_mem f hs).symm


/-- Given a finite set `s`, the product of `f i / g i` over `i ∈ s` equals the product of `f i`
over `i ∈ s` divided by the product of `g i` over `i ∈ s`. -/
@[to_additive
      "Given a finite set `s`, the sum of `f i / g i` over `i ∈ s` equals the sum of `f i`
      over `i ∈ s` minus the sum of `g i` over `i ∈ s`."]
theorem finprod_mem_div_distrib [DivisionCommMonoid G] (f g : α → G) (hs : s.Finite) :
    ∏ᶠ i ∈ s, f i / g i = (∏ᶠ i ∈ s, f i) / ∏ᶠ i ∈ s, g i := by
  /-
    α : Type u_1
    G : Type u_4
    s : Set α
    inst✝ : DivisionCommMonoid G
    f g : α → G
    hs : s.Finite
    ⊢ Eq (finprod fun i => finprod fun h => HDiv.hDiv (f i) (g i)) (HDiv.hDiv (fin …
  -/
  simp only [div_eq_mul_inv, finprod_mem_mul_distrib hs, finprod_mem_inv_distrib g hs]
  /-
    🎉 no goals
  -/


/-- The product of any function over an empty set is `1`. -/
@[to_additive "The sum of any function over an empty set is `0`."]
                                                                /-
                                                                  α : Type u_1
                                                                  M : Type u_5
                                                                  inst✝ : CommMonoid M
                                                                  f : α → M
                                                                  ⊢ Eq (finprod fun i => finprod fun h => f i) 1
                                                                -/
theorem finprod_mem_empty : (∏ᶠ i ∈ (∅ : Set α), f i) = 1 := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- A set `s` is nonempty if the product of some function over `s` is not equal to `1`. -/
@[to_additive "A set `s` is nonempty if the sum of some function over `s` is not equal to `0`."]
theorem nonempty_of_finprod_mem_ne_one (h : ∏ᶠ i ∈ s, f i ≠ 1) : s.Nonempty :=
  nonempty_iff_ne_empty.2 fun h' => h <| h'.symm ▸ finprod_mem_empty


/-- Given finite sets `s` and `t`, the product of `f i` over `i ∈ s ∪ t` times the product of
`f i` over `i ∈ s ∩ t` equals the product of `f i` over `i ∈ s` times the product of `f i`
over `i ∈ t`. -/
@[to_additive
      "Given finite sets `s` and `t`, the sum of `f i` over `i ∈ s ∪ t` plus the sum of
      `f i` over `i ∈ s ∩ t` equals the sum of `f i` over `i ∈ s` plus the sum of `f i`
      over `i ∈ t`."]
theorem finprod_mem_union_inter (hs : s.Finite) (ht : t.Finite) :
    ((∏ᶠ i ∈ s ∪ t, f i) * ∏ᶠ i ∈ s ∩ t, f i) = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    hs : s.Finite
    ht : t.Finite
    ⊢ Eq (HMul.hMul (finprod fun i => finprod fun h => f i) (finprod fun i => finp …
  -/
  lift s to Finset α using hs; lift t to Finset α using ht
  classical
    rw [← Finset.coe_union, ← Finset.coe_inter]
    simp only [finprod_mem_coe_finset, Finset.prod_union_inter]


/-- A more general version of `finprod_mem_union_inter` that requires `s ∩ mulSupport f` and
`t ∩ mulSupport f` rather than `s` and `t` to be finite. -/
@[to_additive
      "A more general version of `finsum_mem_union_inter` that requires `s ∩ support f` and
      `t ∩ support f` rather than `s` and `t` to be finite."]
theorem finprod_mem_union_inter' (hs : (s ∩ mulSupport f).Finite) (ht : (t ∩ mulSupport f).Finite) :
    ((∏ᶠ i ∈ s ∪ t, f i) * ∏ᶠ i ∈ s ∩ t, f i) = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t, f i := by
  rw [← finprod_mem_inter_mulSupport f s, ← finprod_mem_inter_mulSupport f t, ←
    finprod_mem_union_inter hs ht, ← union_inter_distrib_right, finprod_mem_inter_mulSupport, ←
    finprod_mem_inter_mulSupport f (s ∩ t)]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    hs : (Inter.inter s (Function.mulSupport f)).Finite
    ht : (Inter.inter t (Function.mulSupport f)).Finite
    ⊢ Eq (HMul.hMul (finprod fun i => finprod fun h => f i) (finprod fun i => finp …
  -/
  congr 2
  /-
    case e_a.e_f
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    hs : (Inter.inter s (Function.mulSupport f)).Finite
    ht : (Inter.inter t (Function.mulSupport f)).Finite
    ⊢ Eq (fun i => finprod fun h => f i) fun i => finprod fun h => f i
  -/
  rw [inter_left_comm, inter_assoc, inter_assoc, inter_self, inter_left_comm]
  /-
    🎉 no goals
  -/


/-- A more general version of `finprod_mem_union` that requires `s ∩ mulSupport f` and
`t ∩ mulSupport f` rather than `s` and `t` to be finite. -/
@[to_additive
      "A more general version of `finsum_mem_union` that requires `s ∩ support f` and
      `t ∩ support f` rather than `s` and `t` to be finite."]
theorem finprod_mem_union' (hst : Disjoint s t) (hs : (s ∩ mulSupport f).Finite)
    (ht : (t ∩ mulSupport f).Finite) : ∏ᶠ i ∈ s ∪ t, f i = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t, f i := by
  rw [← finprod_mem_union_inter' hs ht, disjoint_iff_inter_eq_empty.1 hst, finprod_mem_empty,
    mul_one]


/-- Given two finite disjoint sets `s` and `t`, the product of `f i` over `i ∈ s ∪ t` equals the
product of `f i` over `i ∈ s` times the product of `f i` over `i ∈ t`. -/
@[to_additive
      "Given two finite disjoint sets `s` and `t`, the sum of `f i` over `i ∈ s ∪ t` equals
      the sum of `f i` over `i ∈ s` plus the sum of `f i` over `i ∈ t`."]
theorem finprod_mem_union (hst : Disjoint s t) (hs : s.Finite) (ht : t.Finite) :
    ∏ᶠ i ∈ s ∪ t, f i = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t, f i :=
  finprod_mem_union' hst (hs.inter_of_left _) (ht.inter_of_left _)


/-- A more general version of `finprod_mem_union'` that requires `s ∩ mulSupport f` and
`t ∩ mulSupport f` rather than `s` and `t` to be disjoint -/
@[to_additive
      "A more general version of `finsum_mem_union'` that requires `s ∩ support f` and
      `t ∩ support f` rather than `s` and `t` to be disjoint"]
theorem finprod_mem_union'' (hst : Disjoint (s ∩ mulSupport f) (t ∩ mulSupport f))
    (hs : (s ∩ mulSupport f).Finite) (ht : (t ∩ mulSupport f).Finite) :
    ∏ᶠ i ∈ s ∪ t, f i = (∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t, f i := by
  rw [← finprod_mem_inter_mulSupport f s, ← finprod_mem_inter_mulSupport f t, ←
    finprod_mem_union hst hs ht, ← union_inter_distrib_right, finprod_mem_inter_mulSupport]


/-- The product of `f i` over `i ∈ {a}` equals `f a`. -/
@[to_additive "The sum of `f i` over `i ∈ {a}` equals `f a`."]
theorem finprod_mem_singleton : (∏ᶠ i ∈ ({a} : Set α), f i) = f a := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    a : α
    ⊢ Eq (finprod fun i => finprod fun h => f i) (f a)
  -/
  rw [← Finset.coe_singleton, finprod_mem_coe_finset, Finset.prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem finprod_cond_eq_left : (∏ᶠ (i) (_ : i = a), f i) = f a :=
  finprod_mem_singleton


@[to_additive (attr := simp)]
                                                                      /-
                                                                        α : Type u_1
                                                                        M : Type u_5
                                                                        inst✝ : CommMonoid M
                                                                        f : α → M
                                                                        a : α
                                                                        ⊢ Eq (finprod fun i => finprod fun x => f i) (f a)
                                                                      -/
theorem finprod_cond_eq_right : (∏ᶠ (i) (_ : a = i), f i) = f a := by simp [@eq_comm _ a]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- A more general version of `finprod_mem_insert` that requires `s ∩ mulSupport f` rather than `s`
to be finite. -/
@[to_additive
      "A more general version of `finsum_mem_insert` that requires `s ∩ support f` rather
      than `s` to be finite."]
theorem finprod_mem_insert' (f : α → M) (h : a ∉ s) (hs : (s ∩ mulSupport f).Finite) :
    ∏ᶠ i ∈ insert a s, f i = f a * ∏ᶠ i ∈ s, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    a : α
    s : Set α
    f : α → M
    h : Not (Membership.mem s a)
    hs : (Inter.inter s (Function.mulSupport f)).Finite
    ⊢ Eq (finprod fun i => finprod fun h => f i) (HMul.hMul (f a) (finprod fun i = …
  -/
  rw [insert_eq, finprod_mem_union' _ _ hs, finprod_mem_singleton]
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      a : α
      s : Set α
      f : α → M
      h : Not (Membership.mem s a)
      hs : (Inter.inter s (Function.mulSupport f)).Finite
      ⊢ Disjoint (Singleton.singleton a) s
    -/
  · rwa [disjoint_singleton_left]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      a : α
      s : Set α
      f : α → M
      h : Not (Membership.mem s a)
      hs : (Inter.inter s (Function.mulSupport f)).Finite
      ⊢ (Inter.inter (Singleton.singleton a) (Function.mulSupport f)).Finite
    -/
  · exact (finite_singleton a).inter_of_left _
    /-
      🎉 no goals
    -/


/-- Given a finite set `s` and an element `a ∉ s`, the product of `f i` over `i ∈ insert a s` equals
`f a` times the product of `f i` over `i ∈ s`. -/
@[to_additive
      "Given a finite set `s` and an element `a ∉ s`, the sum of `f i` over `i ∈ insert a s`
      equals `f a` plus the sum of `f i` over `i ∈ s`."]
theorem finprod_mem_insert (f : α → M) (h : a ∉ s) (hs : s.Finite) :
    ∏ᶠ i ∈ insert a s, f i = f a * ∏ᶠ i ∈ s, f i :=
  finprod_mem_insert' f h <| hs.inter_of_left _


/-- If `f a = 1` when `a ∉ s`, then the product of `f i` over `i ∈ insert a s` equals the product of
`f i` over `i ∈ s`. -/
@[to_additive
      "If `f a = 0` when `a ∉ s`, then the sum of `f i` over `i ∈ insert a s` equals the sum
      of `f i` over `i ∈ s`."]
theorem finprod_mem_insert_of_eq_one_if_not_mem (h : a ∉ s → f a = 1) :
    ∏ᶠ i ∈ insert a s, f i = ∏ᶠ i ∈ s, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    a : α
    s : Set α
    h : Not (Membership.mem s a) → Eq (f a) 1
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun i => finprod fun h = …
  -/
  refine finprod_mem_inter_mulSupport_eq' _ _ _ fun x hx => ⟨?_, Or.inr⟩
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    a : α
    s : Set α
    h : Not (Membership.mem s a) → Eq (f a) 1
    x : α
    hx : Membership.mem (Function.mulSupport f) x
    ⊢ Membership.mem (Insert.insert a s) x → Membership.mem s x
  -/
  rintro (rfl | hxs)
  /-
    case inl
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    x : α
    hx : Membership.mem (Function.mulSupport f) x
    h : Not (Membership.mem s x) → Eq (f x) 1
    ⊢ Membership.mem s x
  -/
  exacts [not_imp_comm.1 h hx, hxs]
  /-
    🎉 no goals
  -/


/-- If `f a = 1`, then the product of `f i` over `i ∈ insert a s` equals the product of `f i` over
`i ∈ s`. -/
@[to_additive
      "If `f a = 0`, then the sum of `f i` over `i ∈ insert a s` equals the sum of `f i`
      over `i ∈ s`."]
theorem finprod_mem_insert_one (h : f a = 1) : ∏ᶠ i ∈ insert a s, f i = ∏ᶠ i ∈ s, f i :=
  finprod_mem_insert_of_eq_one_if_not_mem fun _ => h


/-- If the multiplicative support of `f` is finite, then for every `x` in the domain of `f`, `f x`
divides `finprod f`. -/
theorem finprod_mem_dvd {f : α → N} (a : α) (hf : (mulSupport f).Finite) : f a ∣ finprod f := by
  /-
    α : Type u_1
    N : Type u_6
    inst✝ : CommMonoid N
    f : α → N
    a : α
    hf : (Function.mulSupport f).Finite
    ⊢ Dvd.dvd (f a) (finprod f)
  -/
  by_cases ha : a ∈ mulSupport f
    /-
      case pos
      α : Type u_1
      N : Type u_6
      inst✝ : CommMonoid N
      f : α → N
      a : α
      hf : (Function.mulSupport f).Finite
      ha : Membership.mem (Function.mulSupport f) a
      ⊢ Dvd.dvd (f a) (finprod f)
    -/
  · rw [finprod_eq_prod_of_mulSupport_toFinset_subset f hf (Set.Subset.refl _)]
    /-
      case pos
      α : Type u_1
      N : Type u_6
      inst✝ : CommMonoid N
      f : α → N
      a : α
      hf : (Function.mulSupport f).Finite
      ha : Membership.mem (Function.mulSupport f) a
      ⊢ Dvd.dvd (f a) (hf.toFinset.prod fun i => f i)
    -/
    exact Finset.dvd_prod_of_mem f ((Finite.mem_toFinset hf).mpr ha)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      N : Type u_6
      inst✝ : CommMonoid N
      f : α → N
      a : α
      hf : (Function.mulSupport f).Finite
      ha : Not (Membership.mem (Function.mulSupport f) a)
      ⊢ Dvd.dvd (f a) (finprod f)
    -/
  · rw [nmem_mulSupport.mp ha]
    /-
      case neg
      α : Type u_1
      N : Type u_6
      inst✝ : CommMonoid N
      f : α → N
      a : α
      hf : (Function.mulSupport f).Finite
      ha : Not (Membership.mem (Function.mulSupport f) a)
      ⊢ Dvd.dvd 1 (finprod f)
    -/
    exact one_dvd (finprod f)
    /-
      🎉 no goals
    -/


/-- The product of `f i` over `i ∈ {a, b}`, `a ≠ b`, is equal to `f a * f b`. -/
@[to_additive "The sum of `f i` over `i ∈ {a, b}`, `a ≠ b`, is equal to `f a + f b`."]
theorem finprod_mem_pair (h : a ≠ b) : (∏ᶠ i ∈ ({a, b} : Set α), f i) = f a * f b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    a b : α
    h : Ne a b
    ⊢ Eq (finprod fun i => finprod fun h => f i) (HMul.hMul (f a) (f b))
  -/
  rw [finprod_mem_insert, finprod_mem_singleton]
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    a b : α
    h : Ne a b
    ⊢ Not (Membership.mem (Singleton.singleton b) a)
  -/
  exacts [h, finite_singleton b]
  /-
    🎉 no goals
  -/


/-- The product of `f y` over `y ∈ g '' s` equals the product of `f (g i)` over `s`
provided that `g` is injective on `s ∩ mulSupport (f ∘ g)`. -/
@[to_additive
      "The sum of `f y` over `y ∈ g '' s` equals the sum of `f (g i)` over `s` provided that
      `g` is injective on `s ∩ support (f ∘ g)`."]
theorem finprod_mem_image' {s : Set β} {g : β → α} (hg : (s ∩ mulSupport (f ∘ g)).InjOn g) :
    ∏ᶠ i ∈ g '' s, f i = ∏ᶠ j ∈ s, f (g j) := by
  classical
    by_cases hs : (s ∩ mulSupport (f ∘ g)).Finite
    · have hg : ∀ x ∈ hs.toFinset, ∀ y ∈ hs.toFinset, g x = g y → x = y := by
        simpa only [hs.mem_toFinset]
      have := finprod_mem_eq_prod (comp f g) hs
      unfold Function.comp at this
      rw [this, ← Finset.prod_image hg]
      refine finprod_mem_eq_prod_of_inter_mulSupport_eq f ?_
      rw [Finset.coe_image, hs.coe_toFinset, ← image_inter_mulSupport_eq, inter_assoc, inter_self]
    · unfold Function.comp at hs
      rw [finprod_mem_eq_one_of_infinite hs, finprod_mem_eq_one_of_infinite]
      rwa [image_inter_mulSupport_eq, infinite_image_iff hg]


/-- The product of `f y` over `y ∈ g '' s` equals the product of `f (g i)` over `s` provided that
`g` is injective on `s`. -/
@[to_additive
      "The sum of `f y` over `y ∈ g '' s` equals the sum of `f (g i)` over `s` provided that
      `g` is injective on `s`."]
theorem finprod_mem_image {s : Set β} {g : β → α} (hg : s.InjOn g) :
    ∏ᶠ i ∈ g '' s, f i = ∏ᶠ j ∈ s, f (g j) :=
  finprod_mem_image' <| hg.mono inter_subset_left


/-- The product of `f y` over `y ∈ Set.range g` equals the product of `f (g i)` over all `i`
provided that `g` is injective on `mulSupport (f ∘ g)`. -/
@[to_additive
      "The sum of `f y` over `y ∈ Set.range g` equals the sum of `f (g i)` over all `i`
      provided that `g` is injective on `support (f ∘ g)`."]
theorem finprod_mem_range' {g : β → α} (hg : (mulSupport (f ∘ g)).InjOn g) :
    ∏ᶠ i ∈ range g, f i = ∏ᶠ j, f (g j) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    g : β → α
    hg : Set.InjOn g (Function.mulSupport (Function.comp f g))
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun j => f (g j))
  -/
  rw [← image_univ, finprod_mem_image', finprod_mem_univ]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    g : β → α
    hg : Set.InjOn g (Function.mulSupport (Function.comp f g))
    ⊢ Set.InjOn g (Inter.inter Set.univ (Function.mulSupport (Function.comp f g)))
  -/
  rwa [univ_inter]
  /-
    🎉 no goals
  -/


/-- The product of `f y` over `y ∈ Set.range g` equals the product of `f (g i)` over all `i`
provided that `g` is injective. -/
@[to_additive
      "The sum of `f y` over `y ∈ Set.range g` equals the sum of `f (g i)` over all `i`
      provided that `g` is injective."]
theorem finprod_mem_range {g : β → α} (hg : Injective g) : ∏ᶠ i ∈ range g, f i = ∏ᶠ j, f (g j) :=
  finprod_mem_range' hg.injOn


/-- See also `Finset.prod_bij`. -/
@[to_additive "See also `Finset.sum_bij`."]
theorem finprod_mem_eq_of_bijOn {s : Set α} {t : Set β} {f : α → M} {g : β → M} (e : α → β)
    (he₀ : s.BijOn e t) (he₁ : ∀ x ∈ s, f x = g (e x)) : ∏ᶠ i ∈ s, f i = ∏ᶠ j ∈ t, g j := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Set α
    t : Set β
    f : α → M
    g : β → M
    e : α → β
    he₀ : Set.BijOn e s t
    he₁ : ∀ (x : α), Membership.mem s x → Eq (f x) (g (e x))
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun j => finprod fun h = …
  -/
  rw [← Set.BijOn.image_eq he₀, finprod_mem_image he₀.2.1]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Set α
    t : Set β
    f : α → M
    g : β → M
    e : α → β
    he₀ : Set.BijOn e s t
    he₁ : ∀ (x : α), Membership.mem s x → Eq (f x) (g (e x))
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun j => finprod fun h = …
  -/
  exact finprod_mem_congr rfl he₁
  /-
    🎉 no goals
  -/


/-- See `finprod_comp`, `Fintype.prod_bijective` and `Finset.prod_bij`. -/
@[to_additive "See `finsum_comp`, `Fintype.sum_bijective` and `Finset.sum_bij`."]
theorem finprod_eq_of_bijective {f : α → M} {g : β → M} (e : α → β) (he₀ : Bijective e)
    (he₁ : ∀ x, f x = g (e x)) : ∏ᶠ i, f i = ∏ᶠ j, g j := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    g : β → M
    e : α → β
    he₀ : Function.Bijective e
    he₁ : ∀ (x : α), Eq (f x) (g (e x))
    ⊢ Eq (finprod fun i => f i) (finprod fun j => g j)
  -/
  rw [← finprod_mem_univ f, ← finprod_mem_univ g]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    g : β → M
    e : α → β
    he₀ : Function.Bijective e
    he₁ : ∀ (x : α), Eq (f x) (g (e x))
    ⊢ Eq (finprod fun i => finprod fun h => f i) (finprod fun i => finprod fun h = …
  -/
  exact finprod_mem_eq_of_bijOn _ (bijective_iff_bijOn_univ.mp he₀) fun x _ => he₁ x
  /-
    🎉 no goals
  -/


/-- See also `finprod_eq_of_bijective`, `Fintype.prod_bijective` and `Finset.prod_bij`. -/
@[to_additive "See also `finsum_eq_of_bijective`, `Fintype.sum_bijective` and `Finset.sum_bij`."]
theorem finprod_comp {g : β → M} (e : α → β) (he₀ : Function.Bijective e) :
    (∏ᶠ i, g (e i)) = ∏ᶠ j, g j :=
  finprod_eq_of_bijective e he₀ fun _ => rfl


@[to_additive]
theorem finprod_comp_equiv (e : α ≃ β) {f : β → M} : (∏ᶠ i, f (e i)) = ∏ᶠ i', f i' :=
  finprod_comp e e.bijective


@[to_additive]
theorem finprod_set_coe_eq_finprod_mem (s : Set α) : ∏ᶠ j : s, f j = ∏ᶠ i ∈ s, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    ⊢ Eq (finprod fun j => f ↑j) (finprod fun i => finprod fun h => f i)
  -/
  rw [← finprod_mem_range, Subtype.range_coe]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s : Set α
    ⊢ Function.Injective Subtype.val
  -/
  exact Subtype.coe_injective
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_subtype_eq_finprod_cond (p : α → Prop) :
    ∏ᶠ j : Subtype p, f j = ∏ᶠ (i) (_ : p i), f i :=
  finprod_set_coe_eq_finprod_mem { i | p i }


@[to_additive]
theorem finprod_mem_inter_mul_diff' (t : Set α) (h : (s ∩ mulSupport f).Finite) :
    ((∏ᶠ i ∈ s ∩ t, f i) * ∏ᶠ i ∈ s \ t, f i) = ∏ᶠ i ∈ s, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    h : (Inter.inter s (Function.mulSupport f)).Finite
    ⊢ Eq (HMul.hMul (finprod fun i => finprod fun h => f i) (finprod fun i => finp …
  -/
  rw [← finprod_mem_union', inter_union_diff]
    /-
      case hst
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      f : α → M
      s t : Set α
      h : (Inter.inter s (Function.mulSupport f)).Finite
      ⊢ Disjoint (Inter.inter s t) (SDiff.sdiff s t)
    -/
  · rw [disjoint_iff_inf_le]
    /-
      case hst
      α : Type u_1
      M : Type u_5
      inst✝ : CommMonoid M
      f : α → M
      s t : Set α
      h : (Inter.inter s (Function.mulSupport f)).Finite
      ⊢ LE.le (Min.min (Inter.inter s t) (SDiff.sdiff s t)) Bot.bot
    -/
    exact fun x hx => hx.2.2 hx.1.2
    /-
      🎉 no goals
    -/
  /-
    case hs
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    h : (Inter.inter s (Function.mulSupport f)).Finite
    ⊢ (Inter.inter (Inter.inter s t) (Function.mulSupport f)).Finite
  -/
  exacts [h.subset fun x hx => ⟨hx.1.1, hx.2⟩, h.subset fun x hx => ⟨hx.1.1, hx.2⟩]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_mem_inter_mul_diff (t : Set α) (h : s.Finite) :
    ((∏ᶠ i ∈ s ∩ t, f i) * ∏ᶠ i ∈ s \ t, f i) = ∏ᶠ i ∈ s, f i :=
  finprod_mem_inter_mul_diff' _ <| h.inter_of_left _


/-- A more general version of `finprod_mem_mul_diff` that requires `t ∩ mulSupport f` rather than
`t` to be finite. -/
@[to_additive
      "A more general version of `finsum_mem_add_diff` that requires `t ∩ support f` rather
      than `t` to be finite."]
theorem finprod_mem_mul_diff' (hst : s ⊆ t) (ht : (t ∩ mulSupport f).Finite) :
    ((∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t \ s, f i) = ∏ᶠ i ∈ t, f i := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    s t : Set α
    hst : HasSubset.Subset s t
    ht : (Inter.inter t (Function.mulSupport f)).Finite
    ⊢ Eq (HMul.hMul (finprod fun i => finprod fun h => f i) (finprod fun i => finp …
  -/
  rw [← finprod_mem_inter_mul_diff' _ ht, inter_eq_self_of_subset_right hst]
  /-
    🎉 no goals
  -/


/-- Given a finite set `t` and a subset `s` of `t`, the product of `f i` over `i ∈ s`
times the product of `f i` over `t \ s` equals the product of `f i` over `i ∈ t`. -/
@[to_additive
      "Given a finite set `t` and a subset `s` of `t`, the sum of `f i` over `i ∈ s` plus
      the sum of `f i` over `t \\ s` equals the sum of `f i` over `i ∈ t`."]
theorem finprod_mem_mul_diff (hst : s ⊆ t) (ht : t.Finite) :
    ((∏ᶠ i ∈ s, f i) * ∏ᶠ i ∈ t \ s, f i) = ∏ᶠ i ∈ t, f i :=
  finprod_mem_mul_diff' hst (ht.inter_of_left _)


/-- Given a family of pairwise disjoint finite sets `t i` indexed by a finite type, the product of
`f a` over the union `⋃ i, t i` is equal to the product over all indexes `i` of the products of
`f a` over `a ∈ t i`. -/
@[to_additive
      "Given a family of pairwise disjoint finite sets `t i` indexed by a finite type, the
      sum of `f a` over the union `⋃ i, t i` is equal to the sum over all indexes `i` of the
      sums of `f a` over `a ∈ t i`."]
theorem finprod_mem_iUnion [Finite ι] {t : ι → Set α} (h : Pairwise (Disjoint on t))
    (ht : ∀ i, (t i).Finite) : ∏ᶠ a ∈ ⋃ i : ι, t i, f a = ∏ᶠ i, ∏ᶠ a ∈ t i, f a := by
  /-
    α : Type u_1
    ι : Type u_3
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : Finite ι
    t : ι → Set α
    h : Pairwise (Function.onFun Disjoint t)
    ht : ∀ (i : ι), (t i).Finite
    ⊢ Eq (finprod fun a => finprod fun h => f a) (finprod fun i => finprod fun a = …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    α : Type u_1
    ι : Type u_3
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : Finite ι
    t : ι → Set α
    h : Pairwise (Function.onFun Disjoint t)
    ht : ∀ (i : ι), (t i).Finite
    val✝ : Fintype ι
    ⊢ Eq (finprod fun a => finprod fun h => f a) (finprod fun i => finprod fun a = …
  -/
  lift t to ι → Finset α using ht
  classical
    rw [← biUnion_univ, ← Finset.coe_univ, ← Finset.coe_biUnion, finprod_mem_coe_finset,
      Finset.prod_biUnion]
    · simp only [finprod_mem_coe_finset, finprod_eq_prod_of_fintype]
    · exact fun x _ y _ hxy => Finset.disjoint_coe.1 (h hxy)


/-- Given a family of sets `t : ι → Set α`, a finite set `I` in the index type such that all sets
`t i`, `i ∈ I`, are finite, if all `t i`, `i ∈ I`, are pairwise disjoint, then the product of `f a`
over `a ∈ ⋃ i ∈ I, t i` is equal to the product over `i ∈ I` of the products of `f a` over
`a ∈ t i`. -/
@[to_additive
      "Given a family of sets `t : ι → Set α`, a finite set `I` in the index type such that
      all sets `t i`, `i ∈ I`, are finite, if all `t i`, `i ∈ I`, are pairwise disjoint, then the
      sum of `f a` over `a ∈ ⋃ i ∈ I, t i` is equal to the sum over `i ∈ I` of the sums of `f a`
      over `a ∈ t i`."]
theorem finprod_mem_biUnion {I : Set ι} {t : ι → Set α} (h : I.PairwiseDisjoint t) (hI : I.Finite)
    (ht : ∀ i ∈ I, (t i).Finite) : ∏ᶠ a ∈ ⋃ x ∈ I, t x, f a = ∏ᶠ i ∈ I, ∏ᶠ j ∈ t i, f j := by
  /-
    α : Type u_1
    ι : Type u_3
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    I : Set ι
    t : ι → Set α
    h : I.PairwiseDisjoint t
    hI : I.Finite
    ht : ∀ (i : ι), Membership.mem I i → (t i).Finite
    ⊢ Eq (finprod fun a => finprod fun h => f a) (finprod fun i => finprod fun h = …
  -/
  haveI := hI.fintype
  /-
    α : Type u_1
    ι : Type u_3
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    I : Set ι
    t : ι → Set α
    h : I.PairwiseDisjoint t
    hI : I.Finite
    ht : ∀ (i : ι), Membership.mem I i → (t i).Finite
    this : Fintype ↑I
    ⊢ Eq (finprod fun a => finprod fun h => f a) (finprod fun i => finprod fun h = …
  -/
  rw [biUnion_eq_iUnion, finprod_mem_iUnion, ← finprod_set_coe_eq_finprod_mem]
  /-
    case h
    α : Type u_1
    ι : Type u_3
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    I : Set ι
    t : ι → Set α
    h : I.PairwiseDisjoint t
    hI : I.Finite
    ht : ∀ (i : ι), Membership.mem I i → (t i).Finite
    this : Fintype ↑I
    ⊢ Pairwise (Function.onFun Disjoint fun x => t ↑x)
  -/
  exacts [fun x y hxy => h x.2 y.2 (Subtype.coe_injective.ne hxy), fun b => ht b b.2]
  /-
    🎉 no goals
  -/


/-- If `t` is a finite set of pairwise disjoint finite sets, then the product of `f a`
over `a ∈ ⋃₀ t` is the product over `s ∈ t` of the products of `f a` over `a ∈ s`. -/
@[to_additive
      "If `t` is a finite set of pairwise disjoint finite sets, then the sum of `f a` over
      `a ∈ ⋃₀ t` is the sum over `s ∈ t` of the sums of `f a` over `a ∈ s`."]
theorem finprod_mem_sUnion {t : Set (Set α)} (h : t.PairwiseDisjoint id) (ht₀ : t.Finite)
    (ht₁ : ∀ x ∈ t, Set.Finite x) : ∏ᶠ a ∈ ⋃₀ t, f a = ∏ᶠ s ∈ t, ∏ᶠ a ∈ s, f a := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    t : Set (Set α)
    h : t.PairwiseDisjoint id
    ht₀ : t.Finite
    ht₁ : ∀ (x : Set α), Membership.mem t x → x.Finite
    ⊢ Eq (finprod fun a => finprod fun h => f a) (finprod fun s => finprod fun h = …
  -/
  rw [Set.sUnion_eq_biUnion]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → M
    t : Set (Set α)
    h : t.PairwiseDisjoint id
    ht₀ : t.Finite
    ht₁ : ∀ (x : Set α), Membership.mem t x → x.Finite
    ⊢ Eq (finprod fun a => finprod fun h => f a) (finprod fun s => finprod fun h = …
  -/
  exact finprod_mem_biUnion h ht₀ ht₁
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_finprod_cond_ne (a : α) (hf : (mulSupport f).Finite) :
    (f a * ∏ᶠ (i) (_ : i ≠ a), f i) = ∏ᶠ i, f i := by
  classical
    rw [finprod_eq_prod _ hf]
    have h : ∀ x : α, f x ≠ 1 → (x ≠ a ↔ x ∈ hf.toFinset \ {a}) := by
      intro x hx
      rw [Finset.mem_sdiff, Finset.mem_singleton, Finite.mem_toFinset, mem_mulSupport]
      exact ⟨fun h => And.intro hx h, fun h => h.2⟩
    rw [finprod_cond_eq_prod_of_cond_iff f (fun hx => h _ hx), Finset.sdiff_singleton_eq_erase]
    by_cases ha : a ∈ mulSupport f
    · apply Finset.mul_prod_erase _ _ ((Finite.mem_toFinset _).mpr ha)
    · rw [mem_mulSupport, not_not] at ha
      rw [ha, one_mul]
      apply Finset.prod_erase _ ha


/-- If `s : Set α` and `t : Set β` are finite sets, then taking the product over `s` commutes with
taking the product over `t`. -/
@[to_additive
      "If `s : Set α` and `t : Set β` are finite sets, then summing over `s` commutes with
      summing over `t`."]
theorem finprod_mem_comm {s : Set α} {t : Set β} (f : α → β → M) (hs : s.Finite) (ht : t.Finite) :
    (∏ᶠ i ∈ s, ∏ᶠ j ∈ t, f i j) = ∏ᶠ j ∈ t, ∏ᶠ i ∈ s, f i j := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Set α
    t : Set β
    f : α → β → M
    hs : s.Finite
    ht : t.Finite
    ⊢ Eq (finprod fun i => finprod fun h => finprod fun j => finprod fun h => f i  …
  -/
  lift s to Finset α using hs; lift t to Finset β using ht
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → β → M
    s : Finset α
    t : Finset β
    ⊢ Eq (finprod fun i => finprod fun h => finprod fun j => finprod fun h => f i  …
  -/
  simp only [finprod_mem_coe_finset]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : α → β → M
    s : Finset α
    t : Finset β
    ⊢ Eq (s.prod fun i => t.prod fun i_1 => f i i_1) (t.prod fun i => s.prod fun i …
  -/
  exact Finset.prod_comm
  /-
    🎉 no goals
  -/


/-- To prove a property of a finite product, it suffices to prove that the property is
multiplicative and holds on factors. -/
@[to_additive
      "To prove a property of a finite sum, it suffices to prove that the property is
      additive and holds on summands."]
theorem finprod_mem_induction (p : M → Prop) (hp₀ : p 1) (hp₁ : ∀ x y, p x → p y → p (x * y))
    (hp₂ : ∀ x ∈ s, p <| f x) : p (∏ᶠ i ∈ s, f i) :=
  finprod_induction _ hp₀ hp₁ fun x => finprod_induction _ hp₀ hp₁ <| hp₂ x


theorem finprod_cond_nonneg {R : Type*} [OrderedCommSemiring R] {p : α → Prop} {f : α → R}
    (hf : ∀ x, p x → 0 ≤ f x) : 0 ≤ ∏ᶠ (x) (_ : p x), f x :=
  finprod_nonneg fun x => finprod_nonneg <| hf x


@[to_additive]
theorem single_le_finprod {M : Type*} [OrderedCommMonoid M] (i : α) {f : α → M}
    (hf : (mulSupport f).Finite) (h : ∀ j, 1 ≤ f j) : f i ≤ ∏ᶠ j, f j := by
  classical calc
      f i ≤ ∏ j ∈ insert i hf.toFinset, f j :=
        Finset.single_le_prod' (fun j _ => h j) (Finset.mem_insert_self _ _)
      _ = ∏ᶠ j, f j :=
        (finprod_eq_prod_of_mulSupport_toFinset_subset _ hf (Finset.subset_insert _ _)).symm


theorem finprod_eq_zero {M₀ : Type*} [CommMonoidWithZero M₀] (f : α → M₀) (x : α) (hx : f x = 0)
    (hf : (mulSupport f).Finite) : ∏ᶠ x, f x = 0 := by
  /-
    α : Type u_1
    M₀ : Type u_7
    inst✝ : CommMonoidWithZero M₀
    f : α → M₀
    x : α
    hx : Eq (f x) 0
    hf : (Function.mulSupport f).Finite
    ⊢ Eq (finprod fun x => f x) 0
  -/
  nontriviality
  /-
    α : Type u_1
    M₀ : Type u_7
    inst✝ : CommMonoidWithZero M₀
    f : α → M₀
    x : α
    hx : Eq (f x) 0
    hf : (Function.mulSupport f).Finite
    a✝ : Nontrivial M₀
    ⊢ Eq (finprod fun x => f x) 0
  -/
  rw [finprod_eq_prod f hf]
  /-
    α : Type u_1
    M₀ : Type u_7
    inst✝ : CommMonoidWithZero M₀
    f : α → M₀
    x : α
    hx : Eq (f x) 0
    hf : (Function.mulSupport f).Finite
    a✝ : Nontrivial M₀
    ⊢ Eq (hf.toFinset.prod fun i => f i) 0
  -/
  refine Finset.prod_eq_zero (hf.mem_toFinset.2 ?_) hx
  /-
    α : Type u_1
    M₀ : Type u_7
    inst✝ : CommMonoidWithZero M₀
    f : α → M₀
    x : α
    hx : Eq (f x) 0
    hf : (Function.mulSupport f).Finite
    a✝ : Nontrivial M₀
    ⊢ Membership.mem (Function.mulSupport f) x
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_prod_comm (s : Finset β) (f : α → β → M)
    (h : ∀ b ∈ s, (mulSupport fun a => f a b).Finite) :
    (∏ᶠ a : α, ∏ b ∈ s, f a b) = ∏ b ∈ s, ∏ᶠ a : α, f a b := by
  have hU :
    (mulSupport fun a => ∏ b ∈ s, f a b) ⊆
      (s.finite_toSet.biUnion fun b hb => h b (Finset.mem_coe.1 hb)).toFinset := by
    rw [Finite.coe_toFinset]
    intro x hx
    simp only [exists_prop, mem_iUnion, Ne, mem_mulSupport, Finset.mem_coe]
    contrapose! hx
    rw [mem_mulSupport, not_not, Finset.prod_congr rfl hx, Finset.prod_const_one]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Finset β
    f : α → β → M
    h : ∀ (b : β), Membership.mem s b → (Function.mulSupport fun a => f a b).Finite
    hU : HasSubset.Subset (Function.mulSupport fun a => s.prod fun b => f a b) ↑⋯. …
    ⊢ Eq (finprod fun a => s.prod fun b => f a b) (s.prod fun b => finprod fun a = …
  -/
  rw [finprod_eq_prod_of_mulSupport_subset _ hU, Finset.prod_comm]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Finset β
    f : α → β → M
    h : ∀ (b : β), Membership.mem s b → (Function.mulSupport fun a => f a b).Finite
    hU : HasSubset.Subset (Function.mulSupport fun a => s.prod fun b => f a b) ↑⋯. …
    ⊢ Eq (s.prod fun y => ⋯.toFinset.prod fun x => f x y) (s.prod fun b => finprod …
  -/
  refine Finset.prod_congr rfl fun b hb => (finprod_eq_prod_of_mulSupport_subset _ ?_).symm
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Finset β
    f : α → β → M
    h : ∀ (b : β), Membership.mem s b → (Function.mulSupport fun a => f a b).Finite
    hU : HasSubset.Subset (Function.mulSupport fun a => s.prod fun b => f a b) ↑⋯. …
    b : β
    hb : Membership.mem s b
    ⊢ HasSubset.Subset (Function.mulSupport fun x => f x b) ↑⋯.toFinset
  -/
  intro a ha
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Finset β
    f : α → β → M
    h : ∀ (b : β), Membership.mem s b → (Function.mulSupport fun a => f a b).Finite
    hU : HasSubset.Subset (Function.mulSupport fun a => s.prod fun b => f a b) ↑⋯. …
    b : β
    hb : Membership.mem s b
    a : α
    ha : Membership.mem (Function.mulSupport fun x => f x b) a
    ⊢ Membership.mem (↑⋯.toFinset) a
  -/
  simp only [Finite.coe_toFinset, mem_iUnion]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    s : Finset β
    f : α → β → M
    h : ∀ (b : β), Membership.mem s b → (Function.mulSupport fun a => f a b).Finite
    hU : HasSubset.Subset (Function.mulSupport fun a => s.prod fun b => f a b) ↑⋯. …
    b : β
    hb : Membership.mem s b
    a : α
    ha : Membership.mem (Function.mulSupport fun x => f x b) a
    ⊢ Exists fun i => Exists fun i_1 => Membership.mem (Function.mulSupport fun a  …
  -/
  exact ⟨b, hb, ha⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_finprod_comm (s : Finset α) (f : α → β → M) (h : ∀ a ∈ s, (mulSupport (f a)).Finite) :
    (∏ a ∈ s, ∏ᶠ b : β, f a b) = ∏ᶠ b : β, ∏ a ∈ s, f a b :=
  (finprod_prod_comm s (fun b a => f a b) h).symm


theorem mul_finsum {R : Type*} [Semiring R] (f : α → R) (r : R) (h : (support f).Finite) :
    (r * ∑ᶠ a : α, f a) = ∑ᶠ a : α, r * f a :=
  (AddMonoidHom.mulLeft r).map_finsum h


theorem finsum_mul {R : Type*} [Semiring R] (f : α → R) (r : R) (h : (support f).Finite) :
    (∑ᶠ a : α, f a) * r = ∑ᶠ a : α, f a * r :=
  (AddMonoidHom.mulRight r).map_finsum h


@[to_additive]
theorem Finset.mulSupport_of_fiberwise_prod_subset_image [DecidableEq β] (s : Finset α) (f : α → M)
    (g : α → β) : (mulSupport fun b => (s.filter fun a => g a = b).prod f) ⊆ s.image g := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    inst✝ : DecidableEq β
    s : Finset α
    f : α → M
    g : α → β
    ⊢ HasSubset.Subset (Function.mulSupport fun b => (Finset.filter (fun a => Eq ( …
  -/
  simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe, Function.support_subset_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    inst✝ : DecidableEq β
    s : Finset α
    f : α → M
    g : α → β
    ⊢ HasSubset.Subset (Function.mulSupport fun b => (Finset.filter (fun a => Eq ( …
  -/
  intro b h
  suffices (s.filter fun a : α => g a = b).Nonempty by
    simpa only [fiber_nonempty_iff_mem_image, Finset.mem_image, exists_prop]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    inst✝ : DecidableEq β
    s : Finset α
    f : α → M
    g : α → β
    b : β
    h : Membership.mem (Function.mulSupport fun b => (Finset.filter (fun a => Eq ( …
    ⊢ (Finset.filter (fun a => Eq (g a) b) s).Nonempty
  -/
  exact Finset.nonempty_of_prod_ne_one h
  /-
    🎉 no goals
  -/


/-- Note that `b ∈ (s.filter (fun ab => Prod.fst ab = a)).image Prod.snd` iff `(a, b) ∈ s` so
we can simplify the right hand side of this lemma. However the form stated here is more useful for
iterating this lemma, e.g., if we have `f : α × β × γ → M`. -/
@[to_additive
      "Note that `b ∈ (s.filter (fun ab => Prod.fst ab = a)).image Prod.snd` iff `(a, b) ∈ s` so
      we can simplify the right hand side of this lemma. However the form stated here is more
      useful for iterating this lemma, e.g., if we have `f : α × β × γ → M`."]
theorem finprod_mem_finset_product' [DecidableEq α] [DecidableEq β] (s : Finset (α × β))
    (f : α × β → M) :
    (∏ᶠ (ab) (_ : ab ∈ s), f ab) =
      ∏ᶠ (a) (b) (_ : b ∈ (s.filter fun ab => Prod.fst ab = a).image Prod.snd), f (a, b) := by
  have (a) :
      ∏ i ∈ (s.filter fun ab => Prod.fst ab = a).image Prod.snd, f (a, i) =
        (s.filter (Prod.fst · = a)).prod f := by
    refine Finset.prod_nbij' (fun b ↦ (a, b)) Prod.snd ?_ ?_ ?_ ?_ ?_ <;> aesop
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset (Prod α β)
    f : Prod α β → M
    this : ∀ (a : α), Eq ((Finset.image Prod.snd (Finset.filter (fun ab => Eq ab.1 …
    ⊢ Eq (finprod fun ab => finprod fun x => f ab) (finprod fun a => finprod fun b …
  -/
  rw [finprod_mem_finset_eq_prod]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset (Prod α β)
    f : Prod α β → M
    this : ∀ (a : α), Eq ((Finset.image Prod.snd (Finset.filter (fun ab => Eq ab.1 …
    ⊢ Eq (s.prod fun i => f i) (finprod fun a => finprod fun b => finprod fun x => …
  -/
  simp_rw [finprod_mem_finset_eq_prod, this]
  rw [finprod_eq_prod_of_mulSupport_subset _
      (s.mulSupport_of_fiberwise_prod_subset_image f Prod.fst),
    ← Finset.prod_fiberwise_of_maps_to (t := Finset.image Prod.fst s) _ f]
  -- `finish` could close the goal here
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset (Prod α β)
    f : Prod α β → M
    this : ∀ (a : α), Eq ((Finset.image Prod.snd (Finset.filter (fun ab => Eq ab.1 …
    ⊢ ∀ (i : Prod α β), Membership.mem s i → Membership.mem (Finset.image Prod.fst …
  -/
  simp only [Finset.mem_image]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset (Prod α β)
    f : Prod α β → M
    this : ∀ (a : α), Eq ((Finset.image Prod.snd (Finset.filter (fun ab => Eq ab.1 …
    ⊢ ∀ (i : Prod α β), Membership.mem s i → Exists fun a => And (Membership.mem s …
  -/
  exact fun x hx => ⟨x, hx, rfl⟩
  /-
    🎉 no goals
  -/


/-- See also `finprod_mem_finset_product'`. -/
@[to_additive "See also `finsum_mem_finset_product'`."]
theorem finprod_mem_finset_product (s : Finset (α × β)) (f : α × β → M) :
    (∏ᶠ (ab) (_ : ab ∈ s), f ab) = ∏ᶠ (a) (b) (_ : (a, b) ∈ s), f (a, b) := by
  classical
    rw [finprod_mem_finset_product']
    simp


@[to_additive]
theorem finprod_mem_finset_product₃ {γ : Type*} (s : Finset (α × β × γ)) (f : α × β × γ → M) :
    (∏ᶠ (abc) (_ : abc ∈ s), f abc) = ∏ᶠ (a) (b) (c) (_ : (a, b, c) ∈ s), f (a, b, c) := by
  classical
    rw [finprod_mem_finset_product']
    simp_rw [finprod_mem_finset_product']
    simp


@[to_additive]
theorem finprod_curry (f : α × β → M) (hf : (mulSupport f).Finite) :
    ∏ᶠ ab, f ab = ∏ᶠ (a) (b), f (a, b) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : Prod α β → M
    hf : (Function.mulSupport f).Finite
    ⊢ Eq (finprod fun ab => f ab) (finprod fun a => finprod fun b => f { fst := a, …
  -/
  have h₁ : ∀ a, ∏ᶠ _ : a ∈ hf.toFinset, f a = f a := by simp
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : Prod α β → M
    hf : (Function.mulSupport f).Finite
    h₁ : ∀ (a : Prod α β), Eq (finprod fun x => f a) (f a)
    ⊢ Eq (finprod fun ab => f ab) (finprod fun a => finprod fun b => f { fst := a, …
  -/
  have h₂ : ∏ᶠ a, f a = ∏ᶠ (a) (_ : a ∈ hf.toFinset), f a := by simp
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    f : Prod α β → M
    hf : (Function.mulSupport f).Finite
    h₁ : ∀ (a : Prod α β), Eq (finprod fun x => f a) (f a)
    h₂ : Eq (finprod fun a => f a) (finprod fun a => finprod fun x => f a)
    ⊢ Eq (finprod fun ab => f ab) (finprod fun a => finprod fun b => f { fst := a, …
  -/
  simp_rw [h₂, finprod_mem_finset_product, h₁]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_curry₃ {γ : Type*} (f : α × β × γ → M) (h : (mulSupport f).Finite) :
    ∏ᶠ abc, f abc = ∏ᶠ (a) (b) (c), f (a, b, c) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    γ : Type u_7
    f : Prod α (Prod β γ) → M
    h : (Function.mulSupport f).Finite
    ⊢ Eq (finprod fun abc => f abc) (finprod fun a => finprod fun b => finprod fun …
  -/
  rw [finprod_curry f h]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    γ : Type u_7
    f : Prod α (Prod β γ) → M
    h : (Function.mulSupport f).Finite
    ⊢ Eq (finprod fun a => finprod fun b => f { fst := a, snd := b }) (finprod fun …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    γ : Type u_7
    f : Prod α (Prod β γ) → M
    h : (Function.mulSupport f).Finite
    ⊢ Eq (fun a => finprod fun b => f { fst := a, snd := b }) fun a => finprod fun …
  -/
  ext a
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    γ : Type u_7
    f : Prod α (Prod β γ) → M
    h : (Function.mulSupport f).Finite
    a : α
    ⊢ Eq (finprod fun b => f { fst := a, snd := b }) (finprod fun b => finprod fun …
  -/
  rw [finprod_curry]
  /-
    case e_f.h.hf
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : CommMonoid M
    γ : Type u_7
    f : Prod α (Prod β γ) → M
    h : (Function.mulSupport f).Finite
    a : α
    ⊢ (Function.mulSupport fun b => f { fst := a, snd := b }).Finite
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_dmem {s : Set α} [DecidablePred (· ∈ s)] (f : ∀ a : α, a ∈ s → M) :
    (∏ᶠ (a : α) (h : a ∈ s), f a h) = ∏ᶠ (a : α) (_ : a ∈ s), if h' : a ∈ s then f a h' else 1 :=
  finprod_congr fun _ => finprod_congr fun ha => (dif_pos ha).symm


@[to_additive]
theorem finprod_emb_domain' {f : α → β} (hf : Injective f) [DecidablePred (· ∈ Set.range f)]
    (g : α → M) :
    (∏ᶠ b : β, if h : b ∈ Set.range f then g (Classical.choose h) else 1) = ∏ᶠ a : α, g a := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → β
    hf : Function.Injective f
    inst✝ : DecidablePred fun x => Membership.mem (Set.range f) x
    g : α → M
    ⊢ Eq (finprod fun b => dite (Membership.mem (Set.range f) b) (fun h => g (Clas …
  -/
  simp_rw [← finprod_eq_dif]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → β
    hf : Function.Injective f
    inst✝ : DecidablePred fun x => Membership.mem (Set.range f) x
    g : α → M
    ⊢ Eq (finprod fun b => finprod fun h => g (Classical.choose h)) (finprod fun a …
  -/
  rw [finprod_dmem, finprod_mem_range hf, finprod_congr fun a => _]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → β
    hf : Function.Injective f
    inst✝ : DecidablePred fun x => Membership.mem (Set.range f) x
    g : α → M
    ⊢ ∀ (a : α), Eq (dite (Membership.mem (Set.range f) (f a)) (fun h' => g (Class …
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝¹ : CommMonoid M
    f : α → β
    hf : Function.Injective f
    inst✝ : DecidablePred fun x => Membership.mem (Set.range f) x
    g : α → M
    a : α
    ⊢ Eq (dite (Membership.mem (Set.range f) (f a)) (fun h' => g (Classical.choose …
  -/
  rw [dif_pos (Set.mem_range_self a), hf (Classical.choose_spec (Set.mem_range_self a))]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem finprod_emb_domain (f : α ↪ β) [DecidablePred (· ∈ Set.range f)] (g : α → M) :
    (∏ᶠ b : β, if h : b ∈ Set.range f then g (Classical.choose h) else 1) = ∏ᶠ a : α, g a :=
  finprod_emb_domain' f.injective g


