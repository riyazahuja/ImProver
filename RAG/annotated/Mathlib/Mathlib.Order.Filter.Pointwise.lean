/-- `1 : Filter α` is defined as the filter of sets containing `1 : α` in locale `Pointwise`. -/
@[to_additive
      "`0 : Filter α` is defined as the filter of sets containing `0 : α` in locale `Pointwise`."]
protected def instOne : One (Filter α) :=
  ⟨pure 1⟩


@[to_additive (attr := simp)]
theorem mem_one : s ∈ (1 : Filter α) ↔ (1 : α) ∈ s :=
  mem_pure


@[to_additive]
theorem one_mem_one : (1 : Set α) ∈ (1 : Filter α) :=
  mem_pure.2 Set.one_mem_one


@[to_additive (attr := simp)]
theorem pure_one : pure 1 = (1 : Filter α) :=
  rfl


@[to_additive (attr := simp) zero_prod]
theorem one_prod {l : Filter β} : (1 : Filter α) ×ˢ l = map (1, ·) l := pure_prod


@[to_additive (attr := simp) prod_zero]
theorem prod_one {l : Filter β} : l ×ˢ (1 : Filter α) = map (·, 1) l := prod_pure


@[to_additive (attr := simp)]
theorem principal_one : 𝓟 1 = (1 : Filter α) :=
  principal_singleton _


@[to_additive]
theorem one_neBot : (1 : Filter α).NeBot :=
  Filter.pure_neBot


@[to_additive (attr := simp)]
protected theorem map_one' (f : α → β) : (1 : Filter α).map f = pure (f 1) :=
  rfl


@[to_additive (attr := simp)]
theorem le_one_iff : f ≤ 1 ↔ (1 : Set α) ∈ f :=
  le_pure_iff


@[to_additive]
protected theorem NeBot.le_one_iff (h : f.NeBot) : f ≤ 1 ↔ f = 1 :=
  h.le_pure_iff


@[to_additive (attr := simp)]
theorem eventually_one {p : α → Prop} : (∀ᶠ x in 1, p x) ↔ p 1 :=
  eventually_pure


@[to_additive (attr := simp)]
theorem tendsto_one {a : Filter β} {f : β → α} : Tendsto f a 1 ↔ ∀ᶠ x in a, f x = 1 :=
  tendsto_pure


@[to_additive zero_prod_zero]
theorem one_prod_one [One β] : (1 : Filter α) ×ˢ (1 : Filter β) = 1 :=
  prod_pure_pure


@[deprecated (since := "2024-08-16")] alias zero_sum_zero := zero_prod_zero


/-- `pure` as a `OneHom`. -/
@[to_additive "`pure` as a `ZeroHom`."]
def pureOneHom : OneHom α (Filter α) where
  toFun := pure; map_one' := pure_one


@[to_additive (attr := simp)]
theorem coe_pureOneHom : (pureOneHom : α → Filter α) = pure :=
  rfl


@[to_additive (attr := simp)]
theorem pureOneHom_apply (a : α) : pureOneHom a = pure a :=
  rfl


@[to_additive]
protected theorem map_one [FunLike F α β] [OneHomClass F α β] (φ : F) : map φ 1 = 1 := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : One α
    inst✝² : One β
    inst✝¹ : FunLike F α β
    inst✝ : OneHomClass F α β
    φ : F
    ⊢ Eq (Filter.map (⇑φ) 1) 1
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The inverse of a filter is the pointwise preimage under `⁻¹` of its sets. -/
@[to_additive "The negation of a filter is the pointwise preimage under `-` of its sets."]
instance instInv : Inv (Filter α) :=
  ⟨map Inv.inv⟩


@[to_additive (attr := simp)]
protected theorem map_inv : f.map Inv.inv = f⁻¹ :=
  rfl


@[to_additive]
theorem mem_inv : s ∈ f⁻¹ ↔ Inv.inv ⁻¹' s ∈ f :=
  Iff.rfl


@[to_additive]
protected theorem inv_le_inv (hf : f ≤ g) : f⁻¹ ≤ g⁻¹ :=
  map_mono hf


@[to_additive (attr := simp)]
theorem inv_pure : (pure a : Filter α)⁻¹ = pure a⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem inv_eq_bot_iff : f⁻¹ = ⊥ ↔ f = ⊥ :=
  map_eq_bot_iff


@[to_additive (attr := simp)]
theorem neBot_inv_iff : f⁻¹.NeBot ↔ NeBot f :=
  map_neBot_iff _


@[to_additive]
protected theorem NeBot.inv : f.NeBot → f⁻¹.NeBot := fun h => h.map _


@[to_additive neg.instNeBot]
lemma inv.instNeBot [NeBot f] : NeBot f⁻¹ := .inv ‹_›


@[to_additive (attr := simp)]
protected lemma comap_inv : comap Inv.inv f = f⁻¹ :=
  .symm <| map_eq_comap_of_inverse (inv_comp_inv _) (inv_comp_inv _)


@[to_additive]
                                                   /-
                                                     α : Type u_2
                                                     inst✝ : InvolutiveInv α
                                                     f : Filter α
                                                     s : Set α
                                                     hs : Membership.mem f s
                                                     ⊢ Membership.mem (Inv.inv f) (Inv.inv s)
                                                   -/
theorem inv_mem_inv (hs : s ∈ f) : s⁻¹ ∈ f⁻¹ := by rwa [mem_inv, inv_preimage, inv_inv]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Inversion is involutive on `Filter α` if it is on `α`. -/
@[to_additive "Negation is involutive on `Filter α` if it is on `α`."]
protected def instInvolutiveInv : InvolutiveInv (Filter α) :=
  { Filter.instInv with
                                            /-
                                              F : Type u_1
                                              α : Type u_2
                                              β : Type u_3
                                              γ : Type u_4
                                              δ : Type u_5
                                              ε : Type u_6
                                              inst✝ : InvolutiveInv α
                                              f✝ g : Filter α
                                              s : Set α
                                              f : Filter α
                                              ⊢ Eq (Filter.map (Function.comp Inv.inv Inv.inv) f) f
                                            -/
    inv_inv := fun f => map_map.trans <| by rw [inv_involutive.comp_self, map_id] }
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive (attr := simp)]
protected theorem inv_le_inv_iff : f⁻¹ ≤ g⁻¹ ↔ f ≤ g :=
  ⟨fun h => inv_inv f ▸ inv_inv g ▸ Filter.inv_le_inv h, Filter.inv_le_inv⟩


@[to_additive]
                                                    /-
                                                      α : Type u_2
                                                      inst✝ : InvolutiveInv α
                                                      f g : Filter α
                                                      ⊢ Iff (LE.le (Inv.inv f) g) (LE.le f (Inv.inv g))
                                                    -/
theorem inv_le_iff_le_inv : f⁻¹ ≤ g ↔ f ≤ g⁻¹ := by rw [← Filter.inv_le_inv_iff, inv_inv]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive (attr := simp)]
theorem inv_le_self : f⁻¹ ≤ f ↔ f⁻¹ = f :=
  ⟨fun h => h.antisymm <| inv_le_iff_le_inv.1 h, Eq.le⟩


@[to_additive (attr := simp)]
lemma inv_atTop {G : Type*} [OrderedCommGroup G] : (atTop : Filter G)⁻¹ = atBot :=
  (OrderIso.inv G).map_atTop


/-- The filter `f * g` is generated by `{s * t | s ∈ f, t ∈ g}` in locale `Pointwise`. -/
@[to_additive "The filter `f + g` is generated by `{s + t | s ∈ f, t ∈ g}` in locale `Pointwise`."]
protected def instMul : Mul (Filter α) :=
  ⟨/- This is defeq to `map₂ (· * ·) f g`, but the hypothesis unfolds to `t₁ * t₂ ⊆ s` rather
  than all the way to `Set.image2 (· * ·) t₁ t₂ ⊆ s`. -/
  fun f g => { map₂ (· * ·) f g with sets := { s | ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ * t₂ ⊆ s } }⟩


@[to_additive (attr := simp)]
theorem map₂_mul : map₂ (· * ·) f g = f * g :=
  rfl


@[to_additive]
theorem mem_mul : s ∈ f * g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ * t₂ ⊆ s :=
  Iff.rfl


@[to_additive]
theorem mul_mem_mul : s ∈ f → t ∈ g → s * t ∈ f * g :=
  image2_mem_map₂


@[to_additive (attr := simp)]
theorem bot_mul : ⊥ * g = ⊥ :=
  map₂_bot_left


@[to_additive (attr := simp)]
theorem mul_bot : f * ⊥ = ⊥ :=
  map₂_bot_right


@[to_additive (attr := simp)]
theorem mul_eq_bot_iff : f * g = ⊥ ↔ f = ⊥ ∨ g = ⊥ :=
  map₂_eq_bot_iff


@[to_additive (attr := simp)] -- TODO: make this a scoped instance in the `Pointwise` namespace
lemma mul_neBot_iff : (f * g).NeBot ↔ f.NeBot ∧ g.NeBot :=
  map₂_neBot_iff


@[to_additive]
protected theorem NeBot.mul : NeBot f → NeBot g → NeBot (f * g) :=
  NeBot.map₂


@[to_additive]
theorem NeBot.of_mul_left : (f * g).NeBot → f.NeBot :=
  NeBot.of_map₂_left


@[to_additive]
theorem NeBot.of_mul_right : (f * g).NeBot → g.NeBot :=
  NeBot.of_map₂_right


@[to_additive add.instNeBot]
protected lemma mul.instNeBot [NeBot f] [NeBot g] : NeBot (f * g) := .mul ‹_› ‹_›


@[to_additive (attr := simp)]
theorem pure_mul : pure a * g = g.map (a * ·) :=
  map₂_pure_left


@[to_additive (attr := simp)]
theorem mul_pure : f * pure b = f.map (· * b) :=
  map₂_pure_right


@[to_additive]
                                                                          /-
                                                                            α : Type u_2
                                                                            inst✝ : Mul α
                                                                            a b : α
                                                                            ⊢ Eq (HMul.hMul (Pure.pure a) (Pure.pure b)) (Pure.pure (HMul.hMul a b))
                                                                          -/
theorem pure_mul_pure : (pure a : Filter α) * pure b = pure (a * b) := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive (attr := simp)]
theorem le_mul_iff : h ≤ f * g ↔ ∀ ⦃s⦄, s ∈ f → ∀ ⦃t⦄, t ∈ g → s * t ∈ h :=
  le_map₂_iff


@[to_additive]
instance mulLeftMono : MulLeftMono (Filter α) :=
  ⟨fun _ _ _ => map₂_mono_left⟩


@[to_additive]
instance mulRightMono : MulRightMono (Filter α) :=
  ⟨fun _ _ _ => map₂_mono_right⟩


@[to_additive]
protected theorem map_mul [FunLike F α β] [MulHomClass F α β] (m : F) :
    (f₁ * f₂).map m = f₁.map m * f₂.map m :=
  map_map₂_distrib <| map_mul m


/-- `pure` operation as a `MulHom`. -/
@[to_additive "The singleton operation as an `AddHom`."]
def pureMulHom : α →ₙ* Filter α where
  toFun := pure; map_mul' _ _ := pure_mul_pure.symm


@[to_additive (attr := simp)]
theorem coe_pureMulHom : (pureMulHom : α → Filter α) = pure :=
  rfl


@[to_additive (attr := simp)]
theorem pureMulHom_apply (a : α) : pureMulHom a = pure a :=
  rfl


/-- The filter `f / g` is generated by `{s / t | s ∈ f, t ∈ g}` in locale `Pointwise`. -/
@[to_additive "The filter `f - g` is generated by `{s - t | s ∈ f, t ∈ g}` in locale `Pointwise`."]
protected def instDiv : Div (Filter α) :=
  ⟨/- This is defeq to `map₂ (· / ·) f g`, but the hypothesis unfolds to `t₁ / t₂ ⊆ s`
  rather than all the way to `Set.image2 (· / ·) t₁ t₂ ⊆ s`. -/
  fun f g => { map₂ (· / ·) f g with sets := { s | ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ / t₂ ⊆ s } }⟩


@[to_additive (attr := simp)]
theorem map₂_div : map₂ (· / ·) f g = f / g :=
  rfl


@[to_additive]
theorem mem_div : s ∈ f / g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ / t₂ ⊆ s :=
  Iff.rfl


@[to_additive]
theorem div_mem_div : s ∈ f → t ∈ g → s / t ∈ f / g :=
  image2_mem_map₂


@[to_additive (attr := simp)]
theorem bot_div : ⊥ / g = ⊥ :=
  map₂_bot_left


@[to_additive (attr := simp)]
theorem div_bot : f / ⊥ = ⊥ :=
  map₂_bot_right


@[to_additive (attr := simp)]
theorem div_eq_bot_iff : f / g = ⊥ ↔ f = ⊥ ∨ g = ⊥ :=
  map₂_eq_bot_iff


@[to_additive (attr := simp)]
theorem div_neBot_iff : (f / g).NeBot ↔ f.NeBot ∧ g.NeBot :=
  map₂_neBot_iff


@[to_additive]
protected theorem NeBot.div : NeBot f → NeBot g → NeBot (f / g) :=
  NeBot.map₂


@[to_additive]
theorem NeBot.of_div_left : (f / g).NeBot → f.NeBot :=
  NeBot.of_map₂_left


@[to_additive]
theorem NeBot.of_div_right : (f / g).NeBot → g.NeBot :=
  NeBot.of_map₂_right


@[to_additive sub.instNeBot]
lemma div.instNeBot [NeBot f] [NeBot g] : NeBot (f / g) := .div ‹_› ‹_›


@[to_additive (attr := simp)]
theorem pure_div : pure a / g = g.map (a / ·) :=
  map₂_pure_left


@[to_additive (attr := simp)]
theorem div_pure : f / pure b = f.map (· / b) :=
  map₂_pure_right


@[to_additive]
                                                                          /-
                                                                            α : Type u_2
                                                                            inst✝ : Div α
                                                                            a b : α
                                                                            ⊢ Eq (HDiv.hDiv (Pure.pure a) (Pure.pure b)) (Pure.pure (HDiv.hDiv a b))
                                                                          -/
theorem pure_div_pure : (pure a : Filter α) / pure b = pure (a / b) := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive]
protected theorem div_le_div : f₁ ≤ f₂ → g₁ ≤ g₂ → f₁ / g₁ ≤ f₂ / g₂ :=
  map₂_mono


@[to_additive]
protected theorem div_le_div_left : g₁ ≤ g₂ → f / g₁ ≤ f / g₂ :=
  map₂_mono_left


@[to_additive]
protected theorem div_le_div_right : f₁ ≤ f₂ → f₁ / g ≤ f₂ / g :=
  map₂_mono_right


@[to_additive (attr := simp)]
protected theorem le_div_iff : h ≤ f / g ↔ ∀ ⦃s⦄, s ∈ f → ∀ ⦃t⦄, t ∈ g → s / t ∈ h :=
  le_map₂_iff


@[to_additive]
instance covariant_div : CovariantClass (Filter α) (Filter α) (· / ·) (· ≤ ·) :=
  ⟨fun _ _ _ => map₂_mono_left⟩


@[to_additive]
instance covariant_swap_div : CovariantClass (Filter α) (Filter α) (swap (· / ·)) (· ≤ ·) :=
  ⟨fun _ _ _ => map₂_mono_right⟩


/-- Repeated pointwise addition (not the same as pointwise repeated addition!) of a `Filter`. See
Note [pointwise nat action]. -/
protected def instNSMul [Zero α] [Add α] : SMul ℕ (Filter α) :=
  ⟨nsmulRec⟩


/-- Repeated pointwise multiplication (not the same as pointwise repeated multiplication!) of a
`Filter`. See Note [pointwise nat action]. -/
@[to_additive existing]
protected def instNPow [One α] [Mul α] : Pow (Filter α) ℕ :=
  ⟨fun s n => npowRec n s⟩


/-- Repeated pointwise addition/subtraction (not the same as pointwise repeated
addition/subtraction!) of a `Filter`. See Note [pointwise nat action]. -/
protected def instZSMul [Zero α] [Add α] [Neg α] : SMul ℤ (Filter α) :=
  ⟨zsmulRec⟩


/-- Repeated pointwise multiplication/division (not the same as pointwise repeated
multiplication/division!) of a `Filter`. See Note [pointwise nat action]. -/
@[to_additive existing]
protected def instZPow [One α] [Mul α] [Inv α] : Pow (Filter α) ℤ :=
  ⟨fun s n => zpowRec npowRec n s⟩


/-- `Filter α` is a `Semigroup` under pointwise operations if `α` is. -/
@[to_additive "`Filter α` is an `AddSemigroup` under pointwise operations if `α` is."]
protected def semigroup [Semigroup α] : Semigroup (Filter α) where
  mul := (· * ·)
  mul_assoc _ _ _ := map₂_assoc mul_assoc


/-- `Filter α` is a `CommSemigroup` under pointwise operations if `α` is. -/
@[to_additive "`Filter α` is an `AddCommSemigroup` under pointwise operations if `α` is."]
protected def commSemigroup [CommSemigroup α] : CommSemigroup (Filter α) :=
  { Filter.semigroup with mul_comm := fun _ _ => map₂_comm mul_comm }


/-- `Filter α` is a `MulOneClass` under pointwise operations if `α` is. -/
@[to_additive "`Filter α` is an `AddZeroClass` under pointwise operations if `α` is."]
protected def mulOneClass : MulOneClass (Filter α) where
  one := 1
  mul := (· * ·)
  one_mul := map₂_left_identity one_mul
  mul_one := map₂_right_identity mul_one


/-- If `φ : α →* β` then `mapMonoidHom φ` is the monoid homomorphism
`Filter α →* Filter β` induced by `map φ`. -/
@[to_additive "If `φ : α →+ β` then `mapAddMonoidHom φ` is the monoid homomorphism
 `Filter α →+ Filter β` induced by `map φ`."]
def mapMonoidHom [MonoidHomClass F α β] (φ : F) : Filter α →* Filter β where
  toFun := map φ
  map_one' := Filter.map_one φ
  map_mul' _ _ := Filter.map_mul φ

-- The other direction does not hold in general

@[to_additive]
theorem comap_mul_comap_le [MulHomClass F α β] (m : F) {f g : Filter β} :
    f.comap m * g.comap m ≤ (f * g).comap m := fun _ ⟨_, ⟨t₁, ht₁, t₂, ht₂, t₁t₂⟩, mt⟩ =>
  ⟨m ⁻¹' t₁, ⟨t₁, ht₁, Subset.rfl⟩, m ⁻¹' t₂, ⟨t₂, ht₂, Subset.rfl⟩,
    (preimage_mul_preimage_subset _).trans <| (preimage_mono t₁t₂).trans mt⟩


@[to_additive]
theorem Tendsto.mul_mul [MulHomClass F α β] (m : F) {f₁ g₁ : Filter α} {f₂ g₂ : Filter β} :
    Tendsto m f₁ f₂ → Tendsto m g₁ g₂ → Tendsto m (f₁ * g₁) (f₂ * g₂) := fun hf hg =>
  (Filter.map_mul m).trans_le <| mul_le_mul' hf hg


/-- `pure` as a `MonoidHom`. -/
@[to_additive "`pure` as an `AddMonoidHom`."]
def pureMonoidHom : α →* Filter α :=
  { pureMulHom, pureOneHom with }


@[to_additive (attr := simp)]
theorem coe_pureMonoidHom : (pureMonoidHom : α → Filter α) = pure :=
  rfl


@[to_additive (attr := simp)]
theorem pureMonoidHom_apply (a : α) : pureMonoidHom a = pure a :=
  rfl


/-- `Filter α` is a `Monoid` under pointwise operations if `α` is. -/
@[to_additive "`Filter α` is an `AddMonoid` under pointwise operations if `α` is."]
protected def monoid : Monoid (Filter α) :=
  { Filter.mulOneClass, Filter.semigroup, @Filter.instNPow α _ _ with }


@[to_additive]
theorem pow_mem_pow (hs : s ∈ f) : ∀ n : ℕ, s ^ n ∈ f ^ n
  | 0 => by
    /-
      α : Type u_2
      inst✝ : Monoid α
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      ⊢ Membership.mem (HPow.hPow f 0) (HPow.hPow s 0)
    -/
    rw [pow_zero]
    /-
      α : Type u_2
      inst✝ : Monoid α
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      ⊢ Membership.mem 1 (HPow.hPow s 0)
    -/
    exact one_mem_one
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      α : Type u_2
      inst✝ : Monoid α
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      n : Nat
      ⊢ Membership.mem (HPow.hPow f (HAdd.hAdd n 1)) (HPow.hPow s (HAdd.hAdd n 1))
    -/
    rw [pow_succ]
    /-
      α : Type u_2
      inst✝ : Monoid α
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      n : Nat
      ⊢ Membership.mem (HMul.hMul (HPow.hPow f n) f) (HPow.hPow s (HAdd.hAdd n 1))
    -/
    exact mul_mem_mul (pow_mem_pow hs n) hs
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp) nsmul_bot]
theorem bot_pow {n : ℕ} (hn : n ≠ 0) : (⊥ : Filter α) ^ n = ⊥ := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow Bot.bot n) Bot.bot
  -/
  rw [← Nat.sub_one_add_one hn, pow_succ', bot_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_top_of_one_le (hf : 1 ≤ f) : f * ⊤ = ⊤ := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    ⊢ Eq (HMul.hMul f Top.top) Top.top
  -/
  refine top_le_iff.1 fun s => ?_
  /-
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    s : Set α
    ⊢ Membership.mem (HMul.hMul f Top.top) s → Membership.mem Top.top s
  -/
  simp only [mem_mul, mem_top, exists_and_left, exists_eq_left]
  /-
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    s : Set α
    ⊢ (Exists fun t₁ => And (Membership.mem f t₁) (HasSubset.Subset (HMul.hMul t₁  …
  -/
  rintro ⟨t, ht, hs⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    s t : Set α
    ht : Membership.mem f t
    hs : HasSubset.Subset (HMul.hMul t Set.univ) s
    ⊢ Eq s Set.univ
  -/
  rwa [mul_univ_of_one_mem (mem_one.1 <| hf ht), univ_subset_iff] at hs
  /-
    🎉 no goals
  -/


@[to_additive]
theorem top_mul_of_one_le (hf : 1 ≤ f) : ⊤ * f = ⊤ := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    ⊢ Eq (HMul.hMul Top.top f) Top.top
  -/
  refine top_le_iff.1 fun s => ?_
  /-
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    s : Set α
    ⊢ Membership.mem (HMul.hMul Top.top f) s → Membership.mem Top.top s
  -/
  simp only [mem_mul, mem_top, exists_and_left, exists_eq_left]
  /-
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    s : Set α
    ⊢ (Exists fun t₂ => And (Membership.mem f t₂) (HasSubset.Subset (HMul.hMul Set …
  -/
  rintro ⟨t, ht, hs⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝ : Monoid α
    f : Filter α
    hf : LE.le 1 f
    s t : Set α
    ht : Membership.mem f t
    hs : HasSubset.Subset (HMul.hMul Set.univ t) s
    ⊢ Eq s Set.univ
  -/
  rwa [univ_mul_of_one_mem (mem_one.1 <| hf ht), univ_subset_iff] at hs
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem top_mul_top : (⊤ : Filter α) * ⊤ = ⊤ :=
  mul_top_of_one_le le_top


@[to_additive nsmul_top]
theorem top_pow : ∀ {n : ℕ}, n ≠ 0 → (⊤ : Filter α) ^ n = ⊤
  | 0 => fun h => (h rfl).elim
  | 1 => fun _ => pow_one _
                         /-
                           α : Type u_2
                           inst✝ : Monoid α
                           n : Nat
                           x✝ : Ne (HAdd.hAdd n 2) 0
                           ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd n 2)) Top.top
                         -/
  | n + 2 => fun _ => by rw [pow_succ, top_pow n.succ_ne_zero, top_mul_top]
                         /-
                           🎉 no goals
                         -/


@[to_additive]
protected theorem _root_.IsUnit.filter : IsUnit a → IsUnit (pure a : Filter α) :=
  IsUnit.map (pureMonoidHom : α →* Filter α)


/-- `Filter α` is a `CommMonoid` under pointwise operations if `α` is. -/
@[to_additive "`Filter α` is an `AddCommMonoid` under pointwise operations if `α` is."]
protected def commMonoid [CommMonoid α] : CommMonoid (Filter α) :=
  { Filter.mulOneClass, Filter.commSemigroup with }


@[to_additive]
protected theorem mul_eq_one_iff : f * g = 1 ↔ ∃ a b, f = pure a ∧ g = pure b ∧ a * b = 1 := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    f g : Filter α
    ⊢ Iff (Eq (HMul.hMul f g) 1) (Exists fun a => Exists fun b => And (Eq f (Pure. …
  -/
  refine ⟨fun hfg => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : DivisionMonoid α
      f g : Filter α
      hfg : Eq (HMul.hMul f g) 1
      ⊢ Exists fun a => Exists fun b => And (Eq f (Pure.pure a)) (And (Eq g (Pure.pu …
    -/
  · obtain ⟨t₁, h₁, t₂, h₂, h⟩ : (1 : Set α) ∈ f * g := hfg.symm ▸ one_mem_one
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      f g : Filter α
      hfg : Eq (HMul.hMul f g) 1
      t₁ : Set α
      h₁ : Membership.mem f t₁
      t₂ : Set α
      h₂ : Membership.mem g t₂
      h : HasSubset.Subset (HMul.hMul t₁ t₂) 1
      ⊢ Exists fun a => Exists fun b => And (Eq f (Pure.pure a)) (And (Eq g (Pure.pu …
    -/
    have hfg : (f * g).NeBot := hfg.symm.subst one_neBot
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      f g : Filter α
      hfg✝ : Eq (HMul.hMul f g) 1
      t₁ : Set α
      h₁ : Membership.mem f t₁
      t₂ : Set α
      h₂ : Membership.mem g t₂
      h : HasSubset.Subset (HMul.hMul t₁ t₂) 1
      hfg : (HMul.hMul f g).NeBot
      ⊢ Exists fun a => Exists fun b => And (Eq f (Pure.pure a)) (And (Eq g (Pure.pu …
    -/
    rw [(hfg.nonempty_of_mem <| mul_mem_mul h₁ h₂).subset_one_iff, Set.mul_eq_one_iff] at h
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      f g : Filter α
      hfg✝ : Eq (HMul.hMul f g) 1
      t₁ : Set α
      h₁ : Membership.mem f t₁
      t₂ : Set α
      h₂ : Membership.mem g t₂
      h : Exists fun a => Exists fun b => And (Eq t₁ (Singleton.singleton a)) (And ( …
      hfg : (HMul.hMul f g).NeBot
      ⊢ Exists fun a => Exists fun b => And (Eq f (Pure.pure a)) (And (Eq g (Pure.pu …
    -/
    obtain ⟨a, b, rfl, rfl, h⟩ := h
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      f g : Filter α
      hfg✝ : Eq (HMul.hMul f g) 1
      hfg : (HMul.hMul f g).NeBot
      a b : α
      h₁ : Membership.mem f (Singleton.singleton a)
      h : Eq (HMul.hMul a b) 1
      h₂ : Membership.mem g (Singleton.singleton b)
      ⊢ Exists fun a => Exists fun b => And (Eq f (Pure.pure a)) (And (Eq g (Pure.pu …
    -/
    refine ⟨a, b, ?_, ?_, h⟩
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        α : Type u_2
        inst✝ : DivisionMonoid α
        f g : Filter α
        hfg✝ : Eq (HMul.hMul f g) 1
        hfg : (HMul.hMul f g).NeBot
        a b : α
        h₁ : Membership.mem f (Singleton.singleton a)
        h : Eq (HMul.hMul a b) 1
        h₂ : Membership.mem g (Singleton.singleton b)
        ⊢ Eq f (Pure.pure a)
      -/
    · rwa [← hfg.of_mul_left.le_pure_iff, le_pure_iff]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        α : Type u_2
        inst✝ : DivisionMonoid α
        f g : Filter α
        hfg✝ : Eq (HMul.hMul f g) 1
        hfg : (HMul.hMul f g).NeBot
        a b : α
        h₁ : Membership.mem f (Singleton.singleton a)
        h : Eq (HMul.hMul a b) 1
        h₂ : Membership.mem g (Singleton.singleton b)
        ⊢ Eq g (Pure.pure b)
      -/
    · rwa [← hfg.of_mul_right.le_pure_iff, le_pure_iff]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : DivisionMonoid α
      f g : Filter α
      ⊢ (Exists fun a => Exists fun b => And (Eq f (Pure.pure a)) (And (Eq g (Pure.p …
    -/
  · rintro ⟨a, b, rfl, rfl, h⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      a b : α
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (HMul.hMul (Pure.pure a) (Pure.pure b)) 1
    -/
    rw [pure_mul_pure, h, pure_one]
    /-
      🎉 no goals
    -/


/-- `Filter α` is a division monoid under pointwise operations if `α` is. -/
@[to_additive "`Filter α` is a subtraction monoid under pointwise operations if
 `α` is."]
protected def divisionMonoid : DivisionMonoid (Filter α) :=
  { Filter.monoid, Filter.instInvolutiveInv, Filter.instDiv, Filter.instZPow (α := α) with
    mul_inv_rev := fun _ _ => map_map₂_antidistrib mul_inv_rev
    inv_eq_of_mul := fun s t h => by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        ε : Type u_6
        inst✝ : DivisionMonoid α
        f g s t : Filter α
        h : Eq (HMul.hMul s t) 1
        ⊢ Eq (Inv.inv s) t
      -/
      obtain ⟨a, b, rfl, rfl, hab⟩ := Filter.mul_eq_one_iff.1 h
      /-
        case intro.intro.intro.intro
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        ε : Type u_6
        inst✝ : DivisionMonoid α
        f g : Filter α
        a b : α
        hab : Eq (HMul.hMul a b) 1
        h : Eq (HMul.hMul (Pure.pure a) (Pure.pure b)) 1
        ⊢ Eq (Inv.inv (Pure.pure a)) (Pure.pure b)
      -/
      rw [inv_pure, inv_eq_of_mul_eq_one_right hab]
      /-
        🎉 no goals
      -/
    div_eq_mul_inv := fun _ _ => map_map₂_distrib_right div_eq_mul_inv }


@[to_additive]
theorem isUnit_iff : IsUnit f ↔ ∃ a, f = pure a ∧ IsUnit a := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    f : Filter α
    ⊢ Iff (IsUnit f) (Exists fun a => And (Eq f (Pure.pure a)) (IsUnit a))
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : DivisionMonoid α
      f : Filter α
      ⊢ IsUnit f → Exists fun a => And (Eq f (Pure.pure a)) (IsUnit a)
    -/
  · rintro ⟨u, rfl⟩
    /-
      case mp.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Filter α)
      ⊢ Exists fun a => And (Eq (↑u) (Pure.pure a)) (IsUnit a)
    -/
    obtain ⟨a, b, ha, hb, h⟩ := Filter.mul_eq_one_iff.1 u.mul_inv
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Filter α)
      a b : α
      ha : Eq (↑u) (Pure.pure a)
      hb : Eq (↑(Inv.inv u)) (Pure.pure b)
      h : Eq (HMul.hMul a b) 1
      ⊢ Exists fun a => And (Eq (↑u) (Pure.pure a)) (IsUnit a)
    -/
    refine ⟨a, ha, ⟨a, b, h, pure_injective ?_⟩, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Filter α)
      a b : α
      ha : Eq (↑u) (Pure.pure a)
      hb : Eq (↑(Inv.inv u)) (Pure.pure b)
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (Pure.pure (HMul.hMul b a)) (Pure.pure 1)
    -/
    rw [← pure_mul_pure, ← ha, ← hb]
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Filter α)
      a b : α
      ha : Eq (↑u) (Pure.pure a)
      hb : Eq (↑(Inv.inv u)) (Pure.pure b)
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (HMul.hMul ↑(Inv.inv u) ↑u) (Pure.pure 1)
    -/
    exact u.inv_mul
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : DivisionMonoid α
      f : Filter α
      ⊢ (Exists fun a => And (Eq f (Pure.pure a)) (IsUnit a)) → IsUnit f
    -/
  · rintro ⟨a, rfl, ha⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      a : α
      ha : IsUnit a
      ⊢ IsUnit (Pure.pure a)
    -/
    exact ha.filter
    /-
      🎉 no goals
    -/


/-- `Filter α` is a commutative division monoid under pointwise operations if `α` is. -/
@[to_additive subtractionCommMonoid
      "`Filter α` is a commutative subtraction monoid under pointwise operations if `α` is."]
protected def divisionCommMonoid [DivisionCommMonoid α] : DivisionCommMonoid (Filter α) :=
  { Filter.divisionMonoid, Filter.commSemigroup with }


/-- `Filter α` has distributive negation if `α` has. -/
protected def instDistribNeg [Mul α] [HasDistribNeg α] : HasDistribNeg (Filter α) :=
  { Filter.instInvolutiveNeg with
    neg_mul := fun _ _ => map₂_map_left_comm neg_mul
    mul_neg := fun _ _ => map_map₂_right_comm mul_neg }


theorem mul_add_subset : f * (g + h) ≤ f * g + f * h :=
  map₂_distrib_le_left mul_add


theorem add_mul_subset : (f + g) * h ≤ f * h + g * h :=
  map₂_distrib_le_right add_mul


theorem NeBot.mul_zero_nonneg (hf : f.NeBot) : 0 ≤ f * 0 :=
  le_mul_iff.2 fun _ h₁ _ h₂ =>
    let ⟨_, ha⟩ := hf.nonempty_of_mem h₁
    ⟨_, ha, _, h₂, mul_zero _⟩


theorem NeBot.zero_mul_nonneg (hg : g.NeBot) : 0 ≤ 0 * g :=
  le_mul_iff.2 fun _ h₁ _ h₂ =>
    let ⟨_, hb⟩ := hg.nonempty_of_mem h₂
    ⟨_, h₁, _, hb, zero_mul _⟩


@[to_additive (attr := simp 1100)]
protected theorem one_le_div_iff : 1 ≤ f / g ↔ ¬Disjoint f g := by
  /-
    α : Type u_2
    inst✝ : Group α
    f g : Filter α
    ⊢ Iff (LE.le 1 (HDiv.hDiv f g)) (Not (Disjoint f g))
  -/
  refine ⟨fun h hfg => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : Group α
      f g : Filter α
      h : LE.le 1 (HDiv.hDiv f g)
      hfg : Disjoint f g
      ⊢ False
    -/
  · obtain ⟨s, hs, t, ht, hst⟩ := hfg.le_bot (mem_bot : ∅ ∈ ⊥)
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      f g : Filter α
      h : LE.le 1 (HDiv.hDiv f g)
      hfg : Disjoint f g
      s : Set α
      hs : Membership.mem f s
      t : Set α
      ht : Membership.mem g t
      hst : Eq EmptyCollection.emptyCollection (Inter.inter s t)
      ⊢ False
    -/
    exact Set.one_mem_div_iff.1 (h <| div_mem_div hs ht) (disjoint_iff.2 hst.symm)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : Group α
      f g : Filter α
      ⊢ Not (Disjoint f g) → LE.le 1 (HDiv.hDiv f g)
    -/
  · rintro h s ⟨t₁, h₁, t₂, h₂, hs⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      f g : Filter α
      h : Not (Disjoint f g)
      s t₁ : Set α
      h₁ : Membership.mem f t₁
      t₂ : Set α
      h₂ : Membership.mem g t₂
      hs : HasSubset.Subset (HDiv.hDiv t₁ t₂) s
      ⊢ Membership.mem 1 s
    -/
    exact hs (Set.one_mem_div_iff.2 fun ht => h <| disjoint_of_disjoint_of_mem ht h₁ h₂)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem not_one_le_div_iff : ¬1 ≤ f / g ↔ Disjoint f g :=
  Filter.one_le_div_iff.not_left


@[to_additive]
theorem NeBot.one_le_div (h : f.NeBot) : 1 ≤ f / f := by
  /-
    α : Type u_2
    inst✝ : Group α
    f : Filter α
    h : f.NeBot
    ⊢ LE.le 1 (HDiv.hDiv f f)
  -/
  rintro s ⟨t₁, h₁, t₂, h₂, hs⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝ : Group α
    f : Filter α
    h : f.NeBot
    s t₁ : Set α
    h₁ : Membership.mem f t₁
    t₂ : Set α
    h₂ : Membership.mem f t₂
    hs : HasSubset.Subset (HDiv.hDiv t₁ t₂) s
    ⊢ Membership.mem 1 s
  -/
  obtain ⟨a, ha₁, ha₂⟩ := Set.not_disjoint_iff.1 (h.not_disjoint h₁ h₂)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝ : Group α
    f : Filter α
    h : f.NeBot
    s t₁ : Set α
    h₁ : Membership.mem f t₁
    t₂ : Set α
    h₂ : Membership.mem f t₂
    hs : HasSubset.Subset (HDiv.hDiv t₁ t₂) s
    a : α
    ha₁ : Membership.mem t₁ a
    ha₂ : Membership.mem t₂ a
    ⊢ Membership.mem 1 s
  -/
  rw [mem_one, ← div_self' a]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝ : Group α
    f : Filter α
    h : f.NeBot
    s t₁ : Set α
    h₁ : Membership.mem f t₁
    t₂ : Set α
    h₂ : Membership.mem f t₂
    hs : HasSubset.Subset (HDiv.hDiv t₁ t₂) s
    a : α
    ha₁ : Membership.mem t₁ a
    ha₂ : Membership.mem t₂ a
    ⊢ Membership.mem s (HDiv.hDiv a a)
  -/
  exact hs (Set.div_mem_div ha₁ ha₂)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isUnit_pure (a : α) : IsUnit (pure a : Filter α) :=
  (Group.isUnit a).filter


@[simp]
theorem isUnit_iff_singleton : IsUnit f ↔ ∃ a, f = pure a := by
  /-
    α : Type u_2
    inst✝ : Group α
    f : Filter α
    ⊢ Iff (IsUnit f) (Exists fun a => Eq f (Pure.pure a))
  -/
  simp only [isUnit_iff, Group.isUnit, and_true]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_inv' : f⁻¹.map m = (f.map m)⁻¹ :=
  Semiconj.filter_map (map_inv m) f


@[to_additive]
protected theorem Tendsto.inv_inv : Tendsto m f₁ f₂ → Tendsto m f₁⁻¹ f₂⁻¹ := fun hf =>
  (Filter.map_inv' m).trans_le <| Filter.inv_le_inv hf


@[to_additive]
protected theorem map_div : (f / g).map m = f.map m / g.map m :=
  map_map₂_distrib <| map_div m


@[to_additive]
protected theorem Tendsto.div_div (hf : Tendsto m f₁ f₂) (hg : Tendsto m g₁ g₂) :
    Tendsto m (f₁ / g₁) (f₂ / g₂) :=
  (Filter.map_div m).trans_le <| Filter.div_le_div hf hg


theorem NeBot.div_zero_nonneg (hf : f.NeBot) : 0 ≤ f / 0 :=
  Filter.le_div_iff.2 fun _ h₁ _ h₂ =>
    let ⟨_, ha⟩ := hf.nonempty_of_mem h₁
    ⟨_, ha, _, h₂, div_zero _⟩


theorem NeBot.zero_div_nonneg (hg : g.NeBot) : 0 ≤ 0 / g :=
  Filter.le_div_iff.2 fun _ h₁ _ h₂ =>
    let ⟨_, hb⟩ := hg.nonempty_of_mem h₂
    ⟨_, h₁, _, hb, zero_div _⟩


/-- The filter `f • g` is generated by `{s • t | s ∈ f, t ∈ g}` in locale `Pointwise`. -/
@[to_additive "The filter `f +ᵥ g` is generated by `{s +ᵥ t | s ∈ f, t ∈ g}` in locale
 `Pointwise`."]
protected def instSMul : SMul (Filter α) (Filter β) :=
  ⟨/- This is defeq to `map₂ (· • ·) f g`, but the hypothesis unfolds to `t₁ • t₂ ⊆ s`
  rather than all the way to `Set.image2 (· • ·) t₁ t₂ ⊆ s`. -/
  fun f g => { map₂ (· • ·) f g with sets := { s | ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ • t₂ ⊆ s } }⟩


@[to_additive (attr := simp)]
theorem map₂_smul : map₂ (· • ·) f g = f • g :=
  rfl


@[to_additive]
theorem mem_smul : t ∈ f • g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ • t₂ ⊆ t :=
  Iff.rfl


@[to_additive]
theorem smul_mem_smul : s ∈ f → t ∈ g → s • t ∈ f • g :=
  image2_mem_map₂


@[to_additive (attr := simp)]
theorem bot_smul : (⊥ : Filter α) • g = ⊥ :=
  map₂_bot_left


@[to_additive (attr := simp)]
theorem smul_bot : f • (⊥ : Filter β) = ⊥ :=
  map₂_bot_right


@[to_additive (attr := simp)]
theorem smul_eq_bot_iff : f • g = ⊥ ↔ f = ⊥ ∨ g = ⊥ :=
  map₂_eq_bot_iff


@[to_additive (attr := simp)]
theorem smul_neBot_iff : (f • g).NeBot ↔ f.NeBot ∧ g.NeBot :=
  map₂_neBot_iff


@[to_additive]
protected theorem NeBot.smul : NeBot f → NeBot g → NeBot (f • g) :=
  NeBot.map₂


@[to_additive]
theorem NeBot.of_smul_left : (f • g).NeBot → f.NeBot :=
  NeBot.of_map₂_left


@[to_additive]
theorem NeBot.of_smul_right : (f • g).NeBot → g.NeBot :=
  NeBot.of_map₂_right


@[to_additive vadd.instNeBot]
lemma smul.instNeBot [NeBot f] [NeBot g] : NeBot (f • g) := .smul ‹_› ‹_›


@[to_additive (attr := simp)]
theorem pure_smul : (pure a : Filter α) • g = g.map (a • ·) :=
  map₂_pure_left


@[to_additive (attr := simp)]
theorem smul_pure : f • pure b = f.map (· • b) :=
  map₂_pure_right


@[to_additive]
                                                                                        /-
                                                                                          α : Type u_2
                                                                                          β : Type u_3
                                                                                          inst✝ : SMul α β
                                                                                          a : α
                                                                                          b : β
                                                                                          ⊢ Eq (HSMul.hSMul (Pure.pure a) (Pure.pure b)) (Pure.pure (HSMul.hSMul a b))
                                                                                        -/
theorem pure_smul_pure : (pure a : Filter α) • (pure b : Filter β) = pure (a • b) := by simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[to_additive]
theorem smul_le_smul : f₁ ≤ f₂ → g₁ ≤ g₂ → f₁ • g₁ ≤ f₂ • g₂ :=
  map₂_mono


@[to_additive]
theorem smul_le_smul_left : g₁ ≤ g₂ → f • g₁ ≤ f • g₂ :=
  map₂_mono_left


@[to_additive]
theorem smul_le_smul_right : f₁ ≤ f₂ → f₁ • g ≤ f₂ • g :=
  map₂_mono_right


@[to_additive (attr := simp)]
theorem le_smul_iff : h ≤ f • g ↔ ∀ ⦃s⦄, s ∈ f → ∀ ⦃t⦄, t ∈ g → s • t ∈ h :=
  le_map₂_iff


@[to_additive]
instance covariant_smul : CovariantClass (Filter α) (Filter β) (· • ·) (· ≤ ·) :=
  ⟨fun _ _ _ => map₂_mono_left⟩


/-- The filter `f -ᵥ g` is generated by `{s -ᵥ t | s ∈ f, t ∈ g}` in locale `Pointwise`. -/
protected def instVSub : VSub (Filter α) (Filter β) :=
  ⟨/- This is defeq to `map₂ (-ᵥ) f g`, but the hypothesis unfolds to `t₁ -ᵥ t₂ ⊆ s` rather than all
  the way to `Set.image2 (-ᵥ) t₁ t₂ ⊆ s`. -/
  fun f g => { map₂ (· -ᵥ ·) f g with sets := { s | ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ -ᵥ t₂ ⊆ s } }⟩


@[simp]
theorem map₂_vsub : map₂ (· -ᵥ ·) f g = f -ᵥ g :=
  rfl


theorem mem_vsub {s : Set α} : s ∈ f -ᵥ g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ -ᵥ t₂ ⊆ s :=
  Iff.rfl


theorem vsub_mem_vsub : s ∈ f → t ∈ g → s -ᵥ t ∈ f -ᵥ g :=
  image2_mem_map₂


@[simp]
theorem bot_vsub : (⊥ : Filter β) -ᵥ g = ⊥ :=
  map₂_bot_left


@[simp]
theorem vsub_bot : f -ᵥ (⊥ : Filter β) = ⊥ :=
  map₂_bot_right


@[simp]
theorem vsub_eq_bot_iff : f -ᵥ g = ⊥ ↔ f = ⊥ ∨ g = ⊥ :=
  map₂_eq_bot_iff


@[simp]
theorem vsub_neBot_iff : (f -ᵥ g : Filter α).NeBot ↔ f.NeBot ∧ g.NeBot :=
  map₂_neBot_iff


protected theorem NeBot.vsub : NeBot f → NeBot g → NeBot (f -ᵥ g) :=
  NeBot.map₂


theorem NeBot.of_vsub_left : (f -ᵥ g : Filter α).NeBot → f.NeBot :=
  NeBot.of_map₂_left


theorem NeBot.of_vsub_right : (f -ᵥ g : Filter α).NeBot → g.NeBot :=
  NeBot.of_map₂_right


lemma vsub.instNeBot [NeBot f] [NeBot g] : NeBot (f -ᵥ g) := .vsub ‹_› ‹_›


@[simp]
theorem pure_vsub : (pure a : Filter β) -ᵥ g = g.map (a -ᵥ ·) :=
  map₂_pure_left


@[simp]
theorem vsub_pure : f -ᵥ pure b = f.map (· -ᵥ b) :=
  map₂_pure_right


                                                                                          /-
                                                                                            α : Type u_2
                                                                                            β : Type u_3
                                                                                            inst✝ : VSub α β
                                                                                            a b : β
                                                                                            ⊢ Eq (VSub.vsub (Pure.pure a) (Pure.pure b)) (Pure.pure (VSub.vsub a b))
                                                                                          -/
theorem pure_vsub_pure : (pure a : Filter β) -ᵥ pure b = (pure (a -ᵥ b) : Filter α) := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem vsub_le_vsub : f₁ ≤ f₂ → g₁ ≤ g₂ → f₁ -ᵥ g₁ ≤ f₂ -ᵥ g₂ :=
  map₂_mono


theorem vsub_le_vsub_left : g₁ ≤ g₂ → f -ᵥ g₁ ≤ f -ᵥ g₂ :=
  map₂_mono_left


theorem vsub_le_vsub_right : f₁ ≤ f₂ → f₁ -ᵥ g ≤ f₂ -ᵥ g :=
  map₂_mono_right


@[simp]
theorem le_vsub_iff : h ≤ f -ᵥ g ↔ ∀ ⦃s⦄, s ∈ f → ∀ ⦃t⦄, t ∈ g → s -ᵥ t ∈ h :=
  le_map₂_iff


/-- `a • f` is the map of `f` under `a •` in locale `Pointwise`. -/
@[to_additive "`a +ᵥ f` is the map of `f` under `a +ᵥ` in locale `Pointwise`."]
protected def instSMulFilter : SMul α (Filter β) :=
  ⟨fun a => map (a • ·)⟩


@[to_additive (attr := simp)]
protected theorem map_smul : map (fun b => a • b) f = a • f :=
  rfl


@[to_additive]
theorem mem_smul_filter : s ∈ a • f ↔ (a • ·) ⁻¹' s ∈ f := Iff.rfl


@[to_additive]
theorem smul_set_mem_smul_filter : s ∈ f → a • s ∈ a • f :=
  image_mem_map


@[to_additive (attr := simp)]
theorem smul_filter_bot : a • (⊥ : Filter β) = ⊥ :=
  map_bot


@[to_additive (attr := simp)]
theorem smul_filter_eq_bot_iff : a • f = ⊥ ↔ f = ⊥ :=
  map_eq_bot_iff


@[to_additive (attr := simp)]
theorem smul_filter_neBot_iff : (a • f).NeBot ↔ f.NeBot :=
  map_neBot_iff _


@[to_additive]
theorem NeBot.smul_filter : f.NeBot → (a • f).NeBot := fun h => h.map _


@[to_additive]
theorem NeBot.of_smul_filter : (a • f).NeBot → f.NeBot :=
  NeBot.of_map


@[to_additive vadd_filter.instNeBot]
lemma smul_filter.instNeBot [NeBot f] : NeBot (a • f) := .smul_filter ‹_›


@[to_additive]
theorem smul_filter_le_smul_filter (hf : f₁ ≤ f₂) : a • f₁ ≤ a • f₂ :=
  map_mono hf


@[to_additive]
instance covariant_smul_filter : CovariantClass α (Filter β) (· • ·) (· ≤ ·) :=
  ⟨fun _ => @map_mono β β _⟩


@[to_additive]
instance smulCommClass_filter [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass α β (Filter γ) :=
  ⟨fun _ _ _ => map_comm (funext <| smul_comm _ _) _⟩


@[to_additive]
instance smulCommClass_filter' [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass α (Filter β) (Filter γ) :=
  ⟨fun a _ _ => map_map₂_distrib_right <| smul_comm a⟩


@[to_additive]
instance smulCommClass_filter'' [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass (Filter α) β (Filter γ) :=
  haveI := SMulCommClass.symm α β γ
  SMulCommClass.symm _ _ _


@[to_additive]
instance smulCommClass [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass (Filter α) (Filter β) (Filter γ) :=
  ⟨fun _ _ _ => map₂_left_comm smul_comm⟩


@[to_additive vaddAssocClass]
instance isScalarTower [SMul α β] [SMul α γ] [SMul β γ] [IsScalarTower α β γ] :
    IsScalarTower α β (Filter γ) :=
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     ε : Type u_6
                     inst✝³ : SMul α β
                     inst✝² : SMul α γ
                     inst✝¹ : SMul β γ
                     inst✝ : IsScalarTower α β γ
                     a : α
                     b : β
                     f : Filter γ
                     ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) f) (HSMul.hSMul a (HSMul.hSMul b f))
                   -/
  ⟨fun a b f => by simp only [← Filter.map_smul, map_map, smul_assoc]; rfl⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive vaddAssocClass']
instance isScalarTower' [SMul α β] [SMul α γ] [SMul β γ] [IsScalarTower α β γ] :
    IsScalarTower α (Filter β) (Filter γ) :=
  ⟨fun a f g => by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      ε : Type u_6
      inst✝³ : SMul α β
      inst✝² : SMul α γ
      inst✝¹ : SMul β γ
      inst✝ : IsScalarTower α β γ
      a : α
      f : Filter β
      g : Filter γ
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a f) g) (HSMul.hSMul a (HSMul.hSMul f g))
    -/
    refine (map_map₂_distrib_left fun _ _ => ?_).symm
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      ε : Type u_6
      inst✝³ : SMul α β
      inst✝² : SMul α γ
      inst✝¹ : SMul β γ
      inst✝ : IsScalarTower α β γ
      a : α
      f : Filter β
      g : Filter γ
      x✝¹ : β
      x✝ : γ
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul x✝¹ x✝)) (HSMul.hSMul (HSMul.hSMul a x✝¹) x✝)
    -/
    exact (smul_assoc a _ _).symm⟩
    /-
      🎉 no goals
    -/


@[to_additive vaddAssocClass'']
instance isScalarTower'' [SMul α β] [SMul α γ] [SMul β γ] [IsScalarTower α β γ] :
    IsScalarTower (Filter α) (Filter β) (Filter γ) :=
  ⟨fun _ _ _ => map₂_assoc smul_assoc⟩


@[to_additive]
instance isCentralScalar [SMul α β] [SMul αᵐᵒᵖ β] [IsCentralScalar α β] :
    IsCentralScalar α (Filter β) :=
  ⟨fun _ f => (congr_arg fun m => map m f) <| funext fun _ => op_smul_eq_smul _ _⟩


/-- A multiplicative action of a monoid `α` on a type `β` gives a multiplicative action of
`Filter α` on `Filter β`. -/
@[to_additive "An additive action of an additive monoid `α` on a type `β` gives an additive action
 of `Filter α` on `Filter β`"]
protected def mulAction [Monoid α] [MulAction α β] : MulAction (Filter α) (Filter β) where
                                           /-
                                             F : Type u_1
                                             α : Type u_2
                                             β : Type u_3
                                             γ : Type u_4
                                             δ : Type u_5
                                             ε : Type u_6
                                             inst✝¹ : Monoid α
                                             inst✝ : MulAction α β
                                             f : Filter β
                                             ⊢ Eq (Filter.map (fun x2 => HSMul.hSMul 1 x2) f) f
                                           -/
  one_smul f := map₂_pure_left.trans <| by simp_rw [one_smul, map_id']
                                           /-
                                             🎉 no goals
                                           -/
  mul_smul _ _ _ := map₂_assoc mul_smul


/-- A multiplicative action of a monoid on a type `β` gives a multiplicative action on `Filter β`.
-/
@[to_additive "An additive action of an additive monoid on a type `β` gives an additive action on
 `Filter β`."]
protected def mulActionFilter [Monoid α] [MulAction α β] : MulAction α (Filter β) where
                       /-
                         F : Type u_1
                         α : Type u_2
                         β : Type u_3
                         γ : Type u_4
                         δ : Type u_5
                         ε : Type u_6
                         inst✝¹ : Monoid α
                         inst✝ : MulAction α β
                         a b : α
                         f : Filter β
                         ⊢ Eq (HSMul.hSMul (HMul.hMul a b) f) (HSMul.hSMul a (HSMul.hSMul b f))
                       -/
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     ε : Type u_6
                     inst✝¹ : Monoid α
                     inst✝ : MulAction α β
                     f : Filter β
                     ⊢ Eq (HSMul.hSMul 1 f) f
                   -/
  mul_smul a b f := by simp only [← Filter.map_smul, map_map, Function.comp_def, ← mul_smul]
                   /-
                     🎉 no goals
                   -/
                       /-
                         🎉 no goals
                       -/
  one_smul f := by simp only [← Filter.map_smul, one_smul, map_id']


/-- A distributive multiplicative action of a monoid on an additive monoid `β` gives a distributive
multiplicative action on `Filter β`. -/
protected def distribMulActionFilter [Monoid α] [AddMonoid β] [DistribMulAction α β] :
    DistribMulAction α (Filter β) where
  smul_add _ _ _ := map_map₂_distrib <| smul_add _
                                            /-
                                              F : Type u_1
                                              α : Type u_2
                                              β : Type u_3
                                              γ : Type u_4
                                              δ : Type u_5
                                              ε : Type u_6
                                              inst✝² : Monoid α
                                              inst✝¹ : AddMonoid β
                                              inst✝ : DistribMulAction α β
                                              x✝ : α
                                              ⊢ Eq (Pure.pure (HSMul.hSMul x✝ 0)) 0
                                            -/
  smul_zero _ := (map_pure _ _).trans <| by rw [smul_zero, pure_zero]
                                            /-
                                              🎉 no goals
                                            -/


/-- A multiplicative action of a monoid on a monoid `β` gives a multiplicative action on `Set β`. -/
protected def mulDistribMulActionFilter [Monoid α] [Monoid β] [MulDistribMulAction α β] :
    MulDistribMulAction α (Set β) where
  smul_mul _ _ _ := image_image2_distrib <| smul_mul' _
                                            /-
                                              F : Type u_1
                                              α : Type u_2
                                              β : Type u_3
                                              γ : Type u_4
                                              δ : Type u_5
                                              ε : Type u_6
                                              inst✝² : Monoid α
                                              inst✝¹ : Monoid β
                                              inst✝ : MulDistribMulAction α β
                                              x✝ : α
                                              ⊢ Eq (Singleton.singleton (HSMul.hSMul x✝ 1)) 1
                                            -/
  smul_one _ := image_singleton.trans <| by rw [smul_one, singleton_one]
                                            /-
                                              🎉 no goals
                                            -/


theorem NeBot.smul_zero_nonneg (hf : f.NeBot) : 0 ≤ f • (0 : Filter β) :=
  le_smul_iff.2 fun _ h₁ _ h₂ =>
    let ⟨_, ha⟩ := hf.nonempty_of_mem h₁
    ⟨_, ha, _, h₂, smul_zero _⟩


theorem NeBot.zero_smul_nonneg (hg : g.NeBot) : 0 ≤ (0 : Filter α) • g :=
  le_smul_iff.2 fun _ h₁ _ h₂ =>
    let ⟨_, hb⟩ := hg.nonempty_of_mem h₂
    ⟨_, h₁, _, hb, zero_smul _ _⟩


theorem zero_smul_filter_nonpos : (0 : α) • g ≤ 0 := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Zero α
    inst✝¹ : Zero β
    inst✝ : SMulWithZero α β
    g : Filter β
    ⊢ LE.le (HSMul.hSMul 0 g) 0
  -/
  refine fun s hs => mem_smul_filter.2 ?_
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Zero α
    inst✝¹ : Zero β
    inst✝ : SMulWithZero α β
    g : Filter β
    s : Set β
    hs : Membership.mem 0 s
    ⊢ Membership.mem g (Set.preimage (fun x => HSMul.hSMul 0 x) s)
  -/
  convert @univ_mem _ g
  /-
    case h.e'_5
    α : Type u_2
    β : Type u_3
    inst✝² : Zero α
    inst✝¹ : Zero β
    inst✝ : SMulWithZero α β
    g : Filter β
    s : Set β
    hs : Membership.mem 0 s
    ⊢ Eq (Set.preimage (fun x => HSMul.hSMul 0 x) s) Set.univ
  -/
  refine eq_univ_iff_forall.2 fun a => ?_
  /-
    case h.e'_5
    α : Type u_2
    β : Type u_3
    inst✝² : Zero α
    inst✝¹ : Zero β
    inst✝ : SMulWithZero α β
    g : Filter β
    s : Set β
    hs : Membership.mem 0 s
    a : β
    ⊢ Membership.mem (Set.preimage (fun x => HSMul.hSMul 0 x) s) a
  -/
  rwa [mem_preimage, zero_smul]
  /-
    🎉 no goals
  -/


theorem zero_smul_filter (hg : g.NeBot) : (0 : α) • g = 0 :=
  zero_smul_filter_nonpos.antisymm <|
    le_map_iff.2 fun s hs => by
      /-
        α : Type u_2
        β : Type u_3
        inst✝² : Zero α
        inst✝¹ : Zero β
        inst✝ : SMulWithZero α β
        g : Filter β
        hg : g.NeBot
        s : Set β
        hs : Membership.mem g s
        ⊢ Membership.mem 0 (Set.image (fun x => HSMul.hSMul 0 x) s)
      -/
      simp_rw [zero_smul, (hg.nonempty_of_mem hs).image_const]
      /-
        α : Type u_2
        β : Type u_3
        inst✝² : Zero α
        inst✝¹ : Zero β
        inst✝ : SMulWithZero α β
        g : Filter β
        hg : g.NeBot
        s : Set β
        hs : Membership.mem g s
        ⊢ Membership.mem 0 (Singleton.singleton 0)
      -/
      exact zero_mem_zero
      /-
        🎉 no goals
      -/


