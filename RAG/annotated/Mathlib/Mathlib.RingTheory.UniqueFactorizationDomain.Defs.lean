local infixl:50 " ~ᵤ " => Associated


/-- Well-foundedness of the strict version of ∣, which is equivalent to the descending chain
condition on divisibility and to the ascending chain condition on
principal ideals in an integral domain.
  -/
abbrev WfDvdMonoid (α : Type*) [CommMonoidWithZero α] : Prop :=
  IsWellFounded α DvdNotUnit


theorem wellFounded_dvdNotUnit {α : Type*} [CommMonoidWithZero α] [h : WfDvdMonoid α] :
    WellFounded (DvdNotUnit (α := α)) :=
  h.wf


theorem exists_irreducible_factor {a : α} (ha : ¬IsUnit a) (ha0 : a ≠ 0) :
    ∃ i, Irreducible i ∧ i ∣ a :=
  let ⟨b, hs, hr⟩ := wellFounded_dvdNotUnit.has_min { b | b ∣ a ∧ ¬IsUnit b } ⟨a, dvd_rfl, ha⟩
  ⟨b,
    ⟨hs.2, fun c d he =>
      let h := dvd_trans ⟨d, he⟩ hs.1
      or_iff_not_imp_left.2 fun hc =>
        of_not_not fun hd => hr c ⟨h, hc⟩ ⟨ne_zero_of_dvd_ne_zero ha0 h, d, hd, he⟩⟩,
    hs.1⟩


@[elab_as_elim]
theorem induction_on_irreducible {P : α → Prop} (a : α) (h0 : P 0) (hu : ∀ u : α, IsUnit u → P u)
    (hi : ∀ a i : α, a ≠ 0 → Irreducible i → P a → P (i * a)) : P a :=
  haveI := Classical.dec
  wellFounded_dvdNotUnit.fix
    (fun a ih =>
      if ha0 : a = 0 then ha0.substr h0
      else
        if hau : IsUnit a then hu a hau
        else
          let ⟨i, hii, b, hb⟩ := exists_irreducible_factor hau ha0
          let hb0 : b ≠ 0 := ne_zero_of_dvd_ne_zero ha0 ⟨i, mul_comm i b ▸ hb⟩
          hb.symm ▸ hi b i hb0 hii <| ih b ⟨hb0, i, hii.1, mul_comm i b ▸ hb⟩)
    a


theorem exists_factors (a : α) :
    a ≠ 0 → ∃ f : Multiset α, (∀ b ∈ f, Irreducible b) ∧ Associated f.prod a :=
  induction_on_irreducible a (fun h => (h rfl).elim)
    (fun _ hu _ => ⟨0, fun _ h => False.elim (Multiset.not_mem_zero _ h), hu.unit, one_mul _⟩)
    fun a i ha0 hi ih _ =>
    let ⟨s, hs⟩ := ih ha0
    ⟨i ::ₘ s, fun b H => (Multiset.mem_cons.1 H).elim (fun h => h.symm ▸ hi) (hs.1 b), by
      /-
        α : Type u_1
        inst✝¹ : CommMonoidWithZero α
        inst✝ : WfDvdMonoid α
        a✝ a i : α
        ha0 : Ne a 0
        hi : Irreducible i
        ih : Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → Irreducible …
        x✝ : Ne (HMul.hMul i a) 0
        s : Multiset α
        hs : And (∀ (b : α), Membership.mem s b → Irreducible b) (Associated s.prod a)
        ⊢ Associated (Multiset.cons i s).prod (HMul.hMul i a)
      -/
      rw [s.prod_cons i]
      /-
        α : Type u_1
        inst✝¹ : CommMonoidWithZero α
        inst✝ : WfDvdMonoid α
        a✝ a i : α
        ha0 : Ne a 0
        hi : Irreducible i
        ih : Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → Irreducible …
        x✝ : Ne (HMul.hMul i a) 0
        s : Multiset α
        hs : And (∀ (b : α), Membership.mem s b → Irreducible b) (Associated s.prod a)
        ⊢ Associated (HMul.hMul i s.prod) (HMul.hMul i a)
      -/
      exact hs.2.mul_left i⟩
      /-
        🎉 no goals
      -/


theorem not_unit_iff_exists_factors_eq (a : α) (hn0 : a ≠ 0) :
    ¬IsUnit a ↔ ∃ f : Multiset α, (∀ b ∈ f, Irreducible b) ∧ f.prod = a ∧ f ≠ ∅ :=
  ⟨fun hnu => by
    /-
      α : Type u_1
      inst✝¹ : CommMonoidWithZero α
      inst✝ : WfDvdMonoid α
      a : α
      hn0 : Ne a 0
      hnu : Not (IsUnit a)
      ⊢ Exists fun f => And (∀ (b : α), Membership.mem f b → Irreducible b) (And (Eq …
    -/
    obtain ⟨f, hi, u, rfl⟩ := exists_factors a hn0
    /-
      case intro.intro.intro
      α : Type u_1
      inst✝¹ : CommMonoidWithZero α
      inst✝ : WfDvdMonoid α
      f : Multiset α
      hi : ∀ (b : α), Membership.mem f b → Irreducible b
      u : Units α
      hn0 : Ne (HMul.hMul f.prod ↑u) 0
      hnu : Not (IsUnit (HMul.hMul f.prod ↑u))
      ⊢ Exists fun f_1 => And (∀ (b : α), Membership.mem f_1 b → Irreducible b) (And …
    -/
    obtain ⟨b, h⟩ := Multiset.exists_mem_of_ne_zero fun h : f = 0 => hnu <| by simp [h]
    classical
      refine ⟨(f.erase b).cons (b * u), fun a ha => ?_, ?_, Multiset.cons_ne_zero⟩
      · obtain rfl | ha := Multiset.mem_cons.1 ha
        exacts [Associated.irreducible ⟨u, rfl⟩ (hi b h), hi a (Multiset.mem_of_mem_erase ha)]
      · rw [Multiset.prod_cons, mul_comm b, mul_assoc, Multiset.prod_erase h, mul_comm],
    fun ⟨_, hi, he, hne⟩ =>
    let ⟨b, h⟩ := Multiset.exists_mem_of_ne_zero hne
    not_isUnit_of_not_isUnit_dvd (hi b h).not_unit <| he ▸ Multiset.dvd_prod h⟩


theorem isRelPrime_of_no_irreducible_factors {x y : α} (nonzero : ¬(x = 0 ∧ y = 0))
    (H : ∀ z : α, Irreducible z → z ∣ x → ¬z ∣ y) : IsRelPrime x y :=
  isRelPrime_of_no_nonunits_factors nonzero fun _z znu znz zx zy ↦
    have ⟨i, h1, h2⟩ := exists_irreducible_factor znu znz
    H i h1 (h2.trans zx) (h2.trans zy)


/-- unique factorization monoids.

These are defined as `CancelCommMonoidWithZero`s with well-founded strict divisibility
relations, but this is equivalent to more familiar definitions:

Each element (except zero) is uniquely represented as a multiset of irreducible factors.
Uniqueness is only up to associated elements.

Each element (except zero) is non-uniquely represented as a multiset
of prime factors.

To define a UFD using the definition in terms of multisets
of irreducible factors, use the definition `of_existsUnique_irreducible_factors`

To define a UFD using the definition in terms of multisets
of prime factors, use the definition `of_exists_prime_factors`

-/
class UniqueFactorizationMonoid (α : Type*) [CancelCommMonoidWithZero α] extends
    IsWellFounded α DvdNotUnit : Prop where
  protected irreducible_iff_prime : ∀ {a : α}, Irreducible a ↔ Prime a


instance (priority := 100) ufm_of_decomposition_of_wfDvdMonoid
    [CancelCommMonoidWithZero α] [WfDvdMonoid α] [DecompositionMonoid α] :
    UniqueFactorizationMonoid α :=
  { ‹WfDvdMonoid α› with irreducible_iff_prime := irreducible_iff_prime }


@[deprecated ufm_of_decomposition_of_wfDvdMonoid (since := "2024-02-12")]
theorem ufm_of_gcd_of_wfDvdMonoid [CancelCommMonoidWithZero α] [WfDvdMonoid α]
    [DecompositionMonoid α] : UniqueFactorizationMonoid α :=
  ufm_of_decomposition_of_wfDvdMonoid


theorem exists_prime_factors (a : α) :
    a ≠ 0 → ∃ f : Multiset α, (∀ b ∈ f, Prime b) ∧ f.prod ~ᵤ a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → Prime b) (Asso …
  -/
  simp_rw [← UniqueFactorizationMonoid.irreducible_iff_prime]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → Irreducible b) …
  -/
  apply WfDvdMonoid.exists_factors a
  /-
    🎉 no goals
  -/


lemma exists_prime_iff :
    (∃ (p : α), Prime p) ↔ ∃ (x : α), x ≠ 0 ∧ ¬ IsUnit x := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    ⊢ Iff (Exists fun p => Prime p) (Exists fun x => And (Ne x 0) (Not (IsUnit x)))
  -/
  refine ⟨fun ⟨p, hp⟩ ↦ ⟨p, hp.ne_zero, hp.not_unit⟩, fun ⟨x, hx₀, hxu⟩ ↦ ?_⟩
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x✝ : Exists fun x => And (Ne x 0) (Not (IsUnit x))
    x : α
    hx₀ : Ne x 0
    hxu : Not (IsUnit x)
    ⊢ Exists fun p => Prime p
  -/
  obtain ⟨f, hf, -⟩ := WfDvdMonoid.exists_irreducible_factor hxu hx₀
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x✝ : Exists fun x => And (Ne x 0) (Not (IsUnit x))
    x : α
    hx₀ : Ne x 0
    hxu : Not (IsUnit x)
    f : α
    hf : Irreducible f
    ⊢ Exists fun p => Prime p
  -/
  exact ⟨f, UniqueFactorizationMonoid.irreducible_iff_prime.mp hf⟩
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem induction_on_prime {P : α → Prop} (a : α) (h₁ : P 0) (h₂ : ∀ x : α, IsUnit x → P x)
    (h₃ : ∀ a p : α, a ≠ 0 → Prime p → P a → P (p * a)) : P a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h₁ : P 0
    h₂ : ∀ (x : α), IsUnit x → P x
    h₃ : ∀ (a p : α), Ne a 0 → Prime p → P a → P (HMul.hMul p a)
    ⊢ P a
  -/
  simp_rw [← UniqueFactorizationMonoid.irreducible_iff_prime] at h₃
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h₁ : P 0
    h₂ : ∀ (x : α), IsUnit x → P x
    h₃ : ∀ (a p : α), Ne a 0 → Irreducible p → P a → P (HMul.hMul p a)
    ⊢ P a
  -/
  exact WfDvdMonoid.induction_on_irreducible a h₁ h₂ h₃
  /-
    🎉 no goals
  -/


instance : DecompositionMonoid α where
  primal a := by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      ⊢ IsPrimal a
    -/
    obtain rfl | ha := eq_or_ne a 0; · exact isPrimal_zero
                                       /-
                                         🎉 no goals
                                       -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      ha : Ne a 0
      ⊢ IsPrimal a
    -/
    obtain ⟨f, hf, u, rfl⟩ := exists_prime_factors a ha
    /-
      case inr.intro.intro.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      f : Multiset α
      hf : ∀ (b : α), Membership.mem f b → Prime b
      u : Units α
      ha : Ne (HMul.hMul f.prod ↑u) 0
      ⊢ IsPrimal (HMul.hMul f.prod ↑u)
    -/
    exact ((Submonoid.isPrimal α).multiset_prod_mem f (hf · ·|>.isPrimal)).mul u.isUnit.isPrimal
    /-
      🎉 no goals
    -/


open Classical in
/-- Noncomputably determines the multiset of prime factors. -/
noncomputable def factors (a : α) : Multiset α :=
  if h : a = 0 then 0 else Classical.choose (UniqueFactorizationMonoid.exists_prime_factors a h)


theorem factors_prod {a : α} (ane0 : a ≠ 0) : Associated (factors a).prod a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Associated (UniqueFactorizationMonoid.factors a).prod a
  -/
  rw [factors, dif_neg ane0]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ane0 : Ne a 0
    ⊢ Associated (Classical.choose ⋯).prod a
  -/
  exact (Classical.choose_spec (exists_prime_factors a ane0)).2
  /-
    🎉 no goals
  -/


@[simp]
                                                 /-
                                                   α : Type u_1
                                                   inst✝¹ : CancelCommMonoidWithZero α
                                                   inst✝ : UniqueFactorizationMonoid α
                                                   ⊢ Eq (UniqueFactorizationMonoid.factors 0) 0
                                                 -/
theorem factors_zero : factors (0 : α) = 0 := by simp [factors]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem ne_zero_of_mem_factors {p a : α} (h : p ∈ factors a) : a ≠ 0 := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p a : α
    h : Membership.mem (UniqueFactorizationMonoid.factors a) p
    ⊢ Ne a 0
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p : α
    h : Membership.mem (UniqueFactorizationMonoid.factors 0) p
    ⊢ False
  -/
  simp at h
  /-
    🎉 no goals
  -/


theorem dvd_of_mem_factors {p a : α} (h : p ∈ factors a) : p ∣ a :=
  dvd_trans (Multiset.dvd_prod h) (Associated.dvd (factors_prod (ne_zero_of_mem_factors h)))


theorem prime_of_factor {a : α} (x : α) (hx : x ∈ factors a) : Prime x := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a x : α
    hx : Membership.mem (UniqueFactorizationMonoid.factors a) x
    ⊢ Prime x
  -/
  have ane0 := ne_zero_of_mem_factors hx
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a x : α
    hx : Membership.mem (UniqueFactorizationMonoid.factors a) x
    ane0 : Ne a 0
    ⊢ Prime x
  -/
  rw [factors, dif_neg ane0] at hx
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a x : α
    ane0 : Ne a 0
    hx : Membership.mem (Classical.choose ⋯) x
    ⊢ Prime x
  -/
  exact (Classical.choose_spec (UniqueFactorizationMonoid.exists_prime_factors a ane0)).1 x hx
  /-
    🎉 no goals
  -/


theorem irreducible_of_factor {a : α} : ∀ x : α, x ∈ factors a → Irreducible x := fun x h =>
  (prime_of_factor x h).irreducible


