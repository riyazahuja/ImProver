local infixl:50 " ~ᵤ " => Associated


theorem of_wfDvdMonoid_associates (_ : WfDvdMonoid (Associates α)) : WfDvdMonoid α :=
  ⟨(mk_surjective.wellFounded_iff mk_dvdNotUnit_mk_iff.symm).2 wellFounded_dvdNotUnit⟩


instance wfDvdMonoid_associates : WfDvdMonoid (Associates α) :=
  ⟨(mk_surjective.wellFounded_iff mk_dvdNotUnit_mk_iff.symm).1 wellFounded_dvdNotUnit⟩


theorem wellFoundedLT_associates : WellFoundedLT (Associates α) :=
  ⟨Subrelation.wf dvdNotUnit_of_lt wellFounded_dvdNotUnit⟩


@[deprecated wellFoundedLT_associates (since := "2024-09-02")]
theorem wellFounded_associates : WellFounded ((· < ·) : Associates α → Associates α → Prop) :=
  Subrelation.wf dvdNotUnit_of_lt wellFounded_dvdNotUnit


theorem WfDvdMonoid.of_wellFoundedLT_associates [CancelCommMonoidWithZero α]
    (h : WellFoundedLT (Associates α)) : WfDvdMonoid α :=
  WfDvdMonoid.of_wfDvdMonoid_associates
    ⟨by
      /-
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        h : WellFoundedLT (Associates α)
        ⊢ WellFounded DvdNotUnit
      -/
      convert h.wf
      /-
        case h.e'_2.h.h.h.e
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        h : WellFoundedLT (Associates α)
        x✝¹ x✝ : Associates α
        ⊢ Eq DvdNotUnit LT.lt
      -/
      ext
      /-
        case h.e'_2.h.h.h.e.h.h.a
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        h : WellFoundedLT (Associates α)
        x✝³ x✝² x✝¹ x✝ : Associates α
        ⊢ Iff (DvdNotUnit x✝¹ x✝) (LT.lt x✝¹ x✝)
      -/
      exact Associates.dvdNotUnit_iff_lt⟩
      /-
        🎉 no goals
      -/


@[deprecated WfDvdMonoid.of_wellFoundedLT_associates (since := "2024-09-02")]
theorem WfDvdMonoid.of_wellFounded_associates [CancelCommMonoidWithZero α]
    (h : WellFounded ((· < ·) : Associates α → Associates α → Prop)) : WfDvdMonoid α :=
  WfDvdMonoid.of_wfDvdMonoid_associates
    ⟨by
      /-
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        h : WellFounded fun x1 x2 => LT.lt x1 x2
        ⊢ WellFounded DvdNotUnit
      -/
      convert h
      /-
        case h.e'_2.h.h.h.e
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        h : WellFounded fun x1 x2 => LT.lt x1 x2
        x✝¹ x✝ : Associates α
        ⊢ Eq DvdNotUnit LT.lt
      -/
      ext
      /-
        case h.e'_2.h.h.h.e.h.h.a
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        h : WellFounded fun x1 x2 => LT.lt x1 x2
        x✝³ x✝² x✝¹ x✝ : Associates α
        ⊢ Iff (DvdNotUnit x✝¹ x✝) (LT.lt x✝¹ x✝)
      -/
      exact Associates.dvdNotUnit_iff_lt⟩
      /-
        🎉 no goals
      -/


theorem WfDvdMonoid.iff_wellFounded_associates [CancelCommMonoidWithZero α] :
    WfDvdMonoid α ↔ WellFoundedLT (Associates α) :=
      /-
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        ⊢ WfDvdMonoid α → WellFoundedLT (Associates α)
      -/
  ⟨by apply WfDvdMonoid.wellFoundedLT_associates, WfDvdMonoid.of_wellFoundedLT_associates⟩
      /-
        🎉 no goals
      -/


instance Associates.ufm [CancelCommMonoidWithZero α] [UniqueFactorizationMonoid α] :
    UniqueFactorizationMonoid (Associates α) :=
  { (WfDvdMonoid.wfDvdMonoid_associates : WfDvdMonoid (Associates α)) with
    irreducible_iff_prime := by
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : UniqueFactorizationMonoid α
        ⊢ ∀ {a : Associates α}, Iff (Irreducible a) (Prime a)
      -/
      rw [← Associates.irreducible_iff_prime_iff]
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : UniqueFactorizationMonoid α
        ⊢ ∀ (a : α), Iff (Irreducible a) (Prime a)
      -/
      apply UniqueFactorizationMonoid.irreducible_iff_prime }
      /-
        🎉 no goals
      -/


theorem prime_factors_unique [CancelCommMonoidWithZero α] :
    ∀ {f g : Multiset α},
      (∀ x ∈ f, Prime x) → (∀ x ∈ g, Prime x) → f.prod ~ᵤ g.prod → Multiset.Rel Associated f g := by
  classical
  intro f
  induction' f using Multiset.induction_on with p f ih
  · intros g _ hg h
    exact Multiset.rel_zero_left.2 <|
      Multiset.eq_zero_of_forall_not_mem fun x hx =>
        have : IsUnit g.prod := by simpa [associated_one_iff_isUnit] using h.symm
        (hg x hx).not_unit <|
          isUnit_iff_dvd_one.2 <| (Multiset.dvd_prod hx).trans (isUnit_iff_dvd_one.1 this)
  · intros g hf hg hfg
    let ⟨b, hbg, hb⟩ :=
      (exists_associated_mem_of_dvd_prod (hf p (by simp)) fun q hq => hg _ hq) <|
        hfg.dvd_iff_dvd_right.1 (show p ∣ (p ::ₘ f).prod by simp)
    haveI := Classical.decEq α
    rw [← Multiset.cons_erase hbg]
    exact
      Multiset.Rel.cons hb
        (ih (fun q hq => hf _ (by simp [hq]))
          (fun {q} (hq : q ∈ g.erase b) => hg q (Multiset.mem_of_mem_erase hq))
          (Associated.of_mul_left
            (by rwa [← Multiset.prod_cons, ← Multiset.prod_cons, Multiset.cons_erase hbg]) hb
            (hf p (by simp)).ne_zero))


theorem factors_unique {f g : Multiset α} (hf : ∀ x ∈ f, Irreducible x)
    (hg : ∀ x ∈ g, Irreducible x) (h : f.prod ~ᵤ g.prod) : Multiset.Rel Associated f g :=
  prime_factors_unique (fun x hx => UniqueFactorizationMonoid.irreducible_iff_prime.mp (hf x hx))
    (fun x hx => UniqueFactorizationMonoid.irreducible_iff_prime.mp (hg x hx)) h


/-- If an irreducible has a prime factorization,
  then it is an associate of one of its prime factors. -/
theorem prime_factors_irreducible [CancelCommMonoidWithZero α] {a : α} {f : Multiset α}
    (ha : Irreducible a) (pfa : (∀ b ∈ f, Prime b) ∧ f.prod ~ᵤ a) : ∃ p, a ~ᵤ p ∧ f = {p} := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    a : α
    f : Multiset α
    ha : Irreducible a
    pfa : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod a)
    ⊢ Exists fun p => And (Associated a p) (Eq f (Singleton.singleton p))
  -/
  haveI := Classical.decEq α
  refine @Multiset.induction_on _
    (fun g => (g.prod ~ᵤ a) → (∀ b ∈ g, Prime b) → ∃ p, a ~ᵤ p ∧ g = {p}) f ?_ ?_ pfa.2 pfa.1
    /-
      case refine_1
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      f : Multiset α
      ha : Irreducible a
      pfa : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod a)
      this : DecidableEq α
      ⊢ (fun g => Associated g.prod a → (∀ (b : α), Membership.mem g b → Prime b) →  …
    -/
  · intro h; exact (ha.not_unit (associated_one_iff_isUnit.1 (Associated.symm h))).elim
             /-
               🎉 no goals
             -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      f : Multiset α
      ha : Irreducible a
      pfa : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod a)
      this : DecidableEq α
      ⊢ ∀ (a_1 : α) (s : Multiset α), (fun g => Associated g.prod a → (∀ (b : α), Me …
    -/
  · rintro p s _ ⟨u, hu⟩ hs
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      f : Multiset α
      ha : Irreducible a
      pfa : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod a)
      this : DecidableEq α
      p : α
      s : Multiset α
      a✝ : Associated s.prod a → (∀ (b : α), Membership.mem s b → Prime b) → Exists  …
      u : Units α
      hu : Eq (HMul.hMul (Multiset.cons p s).prod ↑u) a
      hs : ∀ (b : α), Membership.mem (Multiset.cons p s) b → Prime b
      ⊢ Exists fun p_1 => And (Associated a p_1) (Eq (Multiset.cons p s) (Singleton. …
    -/
    use p
    have hs0 : s = 0 := by
      by_contra hs0
      obtain ⟨q, hq⟩ := Multiset.exists_mem_of_ne_zero hs0
      apply (hs q (by simp [hq])).2.1
      refine (ha.isUnit_or_isUnit (?_ : _ = p * ↑u * (s.erase q).prod * _)).resolve_left ?_
      · rw [mul_right_comm _ _ q, mul_assoc, ← Multiset.prod_cons, Multiset.cons_erase hq, ← hu,
          mul_comm, mul_comm p _, mul_assoc]
        simp
      apply mt isUnit_of_mul_isUnit_left (mt isUnit_of_mul_isUnit_left _)
      apply (hs p (Multiset.mem_cons_self _ _)).2.1
    /-
      case h
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      f : Multiset α
      ha : Irreducible a
      pfa : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod a)
      this : DecidableEq α
      p : α
      s : Multiset α
      a✝ : Associated s.prod a → (∀ (b : α), Membership.mem s b → Prime b) → Exists  …
      u : Units α
      hu : Eq (HMul.hMul (Multiset.cons p s).prod ↑u) a
      hs : ∀ (b : α), Membership.mem (Multiset.cons p s) b → Prime b
      hs0 : Eq s 0
      ⊢ And (Associated a p) (Eq (Multiset.cons p s) (Singleton.singleton p))
    -/
    simp only [mul_one, Multiset.prod_cons, Multiset.prod_zero, hs0] at *
    /-
      case h
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      f : Multiset α
      ha : Irreducible a
      pfa : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod a)
      this : DecidableEq α
      p : α
      s : Multiset α
      u : Units α
      a✝ : Associated 1 a → (∀ (b : α), Membership.mem 0 b → Prime b) → Exists fun p …
      hu : Eq (HMul.hMul p ↑u) a
      hs : ∀ (b : α), Membership.mem (Multiset.cons p 0) b → Prime b
      hs0 : True
      ⊢ And (Associated a p) (Eq (Multiset.cons p 0) (Singleton.singleton p))
    -/
    exact ⟨Associated.symm ⟨u, hu⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem irreducible_iff_prime_of_existsUnique_irreducible_factors [CancelCommMonoidWithZero α]
    (eif : ∀ a : α, a ≠ 0 → ∃ f : Multiset α, (∀ b ∈ f, Irreducible b) ∧ f.prod ~ᵤ a)
    (uif :
      ∀ f g : Multiset α,
        (∀ x ∈ f, Irreducible x) →
          (∀ x ∈ g, Irreducible x) → f.prod ~ᵤ g.prod → Multiset.Rel Associated f g)
    (p : α) : Irreducible p ↔ Prime p :=
  letI := Classical.decEq α
  ⟨ fun hpi =>
    ⟨hpi.ne_zero, hpi.1, fun a b ⟨x, hx⟩ =>
      if hab0 : a * b = 0 then
                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝ : CancelCommMonoidWithZero α
                                                                       eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
                                                                       uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
                                                                       p : α
                                                                       this : DecidableEq α := Classical.decEq α
                                                                       hpi : Irreducible p
                                                                       a b : α
                                                                       x✝ : Dvd.dvd p (HMul.hMul a b)
                                                                       x : α
                                                                       hx : Eq (HMul.hMul a b) (HMul.hMul p x)
                                                                       hab0 : Eq (HMul.hMul a b) 0
                                                                       ha0 : Eq a 0
                                                                       ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
                                                                     -/
        (eq_zero_or_eq_zero_of_mul_eq_zero hab0).elim (fun ha0 => by simp [ha0]) fun hb0 => by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
          /-
            α : Type u_1
            inst✝ : CancelCommMonoidWithZero α
            eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
            uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
            p : α
            this : DecidableEq α := Classical.decEq α
            hpi : Irreducible p
            a b : α
            x✝ : Dvd.dvd p (HMul.hMul a b)
            x : α
            hx : Eq (HMul.hMul a b) (HMul.hMul p x)
            hab0 : Eq (HMul.hMul a b) 0
            hb0 : Eq b 0
            ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
          -/
          simp [hb0]
          /-
            🎉 no goals
          -/
      else by
        /-
          α : Type u_1
          inst✝ : CancelCommMonoidWithZero α
          eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
          uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
          p : α
          this : DecidableEq α := Classical.decEq α
          hpi : Irreducible p
          a b : α
          x✝ : Dvd.dvd p (HMul.hMul a b)
          x : α
          hx : Eq (HMul.hMul a b) (HMul.hMul p x)
          hab0 : Not (Eq (HMul.hMul a b) 0)
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        have hx0 : x ≠ 0 := fun hx0 => by simp_all
        /-
          α : Type u_1
          inst✝ : CancelCommMonoidWithZero α
          eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
          uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
          p : α
          this : DecidableEq α := Classical.decEq α
          hpi : Irreducible p
          a b : α
          x✝ : Dvd.dvd p (HMul.hMul a b)
          x : α
          hx : Eq (HMul.hMul a b) (HMul.hMul p x)
          hab0 : Not (Eq (HMul.hMul a b) 0)
          hx0 : Ne x 0
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        have ha0 : a ≠ 0 := left_ne_zero_of_mul hab0
        /-
          α : Type u_1
          inst✝ : CancelCommMonoidWithZero α
          eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
          uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
          p : α
          this : DecidableEq α := Classical.decEq α
          hpi : Irreducible p
          a b : α
          x✝ : Dvd.dvd p (HMul.hMul a b)
          x : α
          hx : Eq (HMul.hMul a b) (HMul.hMul p x)
          hab0 : Not (Eq (HMul.hMul a b) 0)
          hx0 : Ne x 0
          ha0 : Ne a 0
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        have hb0 : b ≠ 0 := right_ne_zero_of_mul hab0
        /-
          α : Type u_1
          inst✝ : CancelCommMonoidWithZero α
          eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
          uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
          p : α
          this : DecidableEq α := Classical.decEq α
          hpi : Irreducible p
          a b : α
          x✝ : Dvd.dvd p (HMul.hMul a b)
          x : α
          hx : Eq (HMul.hMul a b) (HMul.hMul p x)
          hab0 : Not (Eq (HMul.hMul a b) 0)
          hx0 : Ne x 0
          ha0 : Ne a 0
          hb0 : Ne b 0
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        cases' eif x hx0 with fx hfx
        /-
          case intro
          α : Type u_1
          inst✝ : CancelCommMonoidWithZero α
          eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
          uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
          p : α
          this : DecidableEq α := Classical.decEq α
          hpi : Irreducible p
          a b : α
          x✝ : Dvd.dvd p (HMul.hMul a b)
          x : α
          hx : Eq (HMul.hMul a b) (HMul.hMul p x)
          hab0 : Not (Eq (HMul.hMul a b) 0)
          hx0 : Ne x 0
          ha0 : Ne a 0
          hb0 : Ne b 0
          fx : Multiset α
          hfx : And (∀ (b : α), Membership.mem fx b → Irreducible b) (Associated fx.prod …
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        cases' eif a ha0 with fa hfa
        /-
          case intro.intro
          α : Type u_1
          inst✝ : CancelCommMonoidWithZero α
          eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
          uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
          p : α
          this : DecidableEq α := Classical.decEq α
          hpi : Irreducible p
          a b : α
          x✝ : Dvd.dvd p (HMul.hMul a b)
          x : α
          hx : Eq (HMul.hMul a b) (HMul.hMul p x)
          hab0 : Not (Eq (HMul.hMul a b) 0)
          hx0 : Ne x 0
          ha0 : Ne a 0
          hb0 : Ne b 0
          fx : Multiset α
          hfx : And (∀ (b : α), Membership.mem fx b → Irreducible b) (Associated fx.prod …
          fa : Multiset α
          hfa : And (∀ (b : α), Membership.mem fa b → Irreducible b) (Associated fa.prod …
          ⊢ Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        cases' eif b hb0 with fb hfb
        have h : Multiset.Rel Associated (p ::ₘ fx) (fa + fb) := by
          apply uif
          · exact fun i hi => (Multiset.mem_cons.1 hi).elim (fun hip => hip.symm ▸ hpi) (hfx.1 _)
          · exact fun i hi => (Multiset.mem_add.1 hi).elim (hfa.1 _) (hfb.1 _)
          calc
            Multiset.prod (p ::ₘ fx) ~ᵤ a * b := by
              rw [hx, Multiset.prod_cons]; exact hfx.2.mul_left _
            _ ~ᵤ fa.prod * fb.prod := hfa.2.symm.mul_mul hfb.2.symm
            _ = _ := by rw [Multiset.prod_add]

        exact
          let ⟨q, hqf, hq⟩ := Multiset.exists_mem_of_rel_of_mem h (Multiset.mem_cons_self p _)
          (Multiset.mem_add.1 hqf).elim
            (fun hqa =>
              Or.inl <| hq.dvd_iff_dvd_left.2 <| hfa.2.dvd_iff_dvd_right.1 (Multiset.dvd_prod hqa))
            fun hqb =>
            Or.inr <| hq.dvd_iff_dvd_left.2 <| hfb.2.dvd_iff_dvd_right.1 (Multiset.dvd_prod hqb)⟩,
    Prime.irreducible⟩


@[deprecated (since := "2024-12-17")]
alias irreducible_iff_prime_of_exists_unique_irreducible_factors :=
  irreducible_iff_prime_of_existsUnique_irreducible_factors


@[simp]
theorem factors_one : factors (1 : α) = 0 := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    ⊢ Eq (UniqueFactorizationMonoid.factors 1) 0
  -/
  nontriviality α using factors
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a✝ : Nontrivial α
    ⊢ Eq (UniqueFactorizationMonoid.factors 1) 0
  -/
  rw [← Multiset.rel_zero_right]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a✝ : Nontrivial α
    ⊢ Multiset.Rel ?m.44160 (UniqueFactorizationMonoid.factors 1) 0
  -/
  refine factors_unique irreducible_of_factor (fun x hx => (Multiset.not_mem_zero x hx).elim) ?_
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a✝ : Nontrivial α
    ⊢ Associated (UniqueFactorizationMonoid.factors 1).prod (Multiset.prod 0)
  -/
  rw [Multiset.prod_zero]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a✝ : Nontrivial α
    ⊢ Associated (UniqueFactorizationMonoid.factors 1).prod 1
  -/
  exact factors_prod one_ne_zero
  /-
    🎉 no goals
  -/


theorem exists_mem_factors_of_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) :
    p ∣ a → ∃ q ∈ factors a, p ~ᵤ q := fun ⟨b, hb⟩ =>
                                    /-
                                      α : Type u_1
                                      inst✝¹ : CancelCommMonoidWithZero α
                                      inst✝ : UniqueFactorizationMonoid α
                                      a p : α
                                      ha0 : Ne a 0
                                      hp : Irreducible p
                                      x✝ : Dvd.dvd p a
                                      b : α
                                      hb : Eq a (HMul.hMul p b)
                                      hb0 : Eq b 0
                                      ⊢ False
                                    -/
  have hb0 : b ≠ 0 := fun hb0 => by simp_all
                                    /-
                                      🎉 no goals
                                    -/
  have : Multiset.Rel Associated (p ::ₘ factors b) (factors a) :=
    factors_unique
      (fun _ hx => (Multiset.mem_cons.1 hx).elim (fun h => h.symm ▸ hp) (irreducible_of_factor _))
      irreducible_of_factor
      (Associated.symm <|
        calc
          Multiset.prod (factors a) ~ᵤ a := factors_prod ha0
          _ = p * b := hb
          _ ~ᵤ Multiset.prod (p ::ₘ factors b) := by
            /-
              α : Type u_1
              inst✝¹ : CancelCommMonoidWithZero α
              inst✝ : UniqueFactorizationMonoid α
              a p : α
              ha0 : Ne a 0
              hp : Irreducible p
              x✝ : Dvd.dvd p a
              b : α
              hb : Eq a (HMul.hMul p b)
              hb0 : Ne b 0
              ⊢ Associated (HMul.hMul p b) (Multiset.cons p (UniqueFactorizationMonoid.facto …
            -/
            rw [Multiset.prod_cons]; exact (factors_prod hb0).symm.mul_left _
                                     /-
                                       🎉 no goals
                                     -/
          )
                                             /-
                                               α : Type u_1
                                               inst✝¹ : CancelCommMonoidWithZero α
                                               inst✝ : UniqueFactorizationMonoid α
                                               a p : α
                                               ha0 : Ne a 0
                                               hp : Irreducible p
                                               x✝ : Dvd.dvd p a
                                               b : α
                                               hb : Eq a (HMul.hMul p b)
                                               hb0 : Ne b 0
                                               this : Multiset.Rel Associated (Multiset.cons p (UniqueFactorizationMonoid.fac …
                                               ⊢ Membership.mem (Multiset.cons p (UniqueFactorizationMonoid.factors b)) p
                                             -/
  Multiset.exists_mem_of_rel_of_mem this (by simp)
                                             /-
                                               🎉 no goals
                                             -/


theorem exists_mem_factors {x : α} (hx : x ≠ 0) (h : ¬IsUnit x) : ∃ p, p ∈ factors x := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    h : Not (IsUnit x)
    ⊢ Exists fun p => Membership.mem (UniqueFactorizationMonoid.factors x) p
  -/
  obtain ⟨p', hp', hp'x⟩ := WfDvdMonoid.exists_irreducible_factor h hx
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    h : Not (IsUnit x)
    p' : α
    hp' : Irreducible p'
    hp'x : Dvd.dvd p' x
    ⊢ Exists fun p => Membership.mem (UniqueFactorizationMonoid.factors x) p
  -/
  obtain ⟨p, hp, _⟩ := exists_mem_factors_of_dvd hx hp' hp'x
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    h : Not (IsUnit x)
    p' : α
    hp' : Irreducible p'
    hp'x : Dvd.dvd p' x
    p : α
    hp : Membership.mem (UniqueFactorizationMonoid.factors x) p
    right✝ : Associated p' p
    ⊢ Exists fun p => Membership.mem (UniqueFactorizationMonoid.factors x) p
  -/
  exact ⟨p, hp⟩
  /-
    🎉 no goals
  -/


open Classical in
theorem factors_mul {x y : α} (hx : x ≠ 0) (hy : y ≠ 0) :
    Multiset.Rel Associated (factors (x * y)) (factors x + factors y) := by
  refine
    factors_unique irreducible_of_factor
      (fun a ha =>
        (Multiset.mem_add.mp ha).by_cases (irreducible_of_factor _) (irreducible_of_factor _))
      ((factors_prod (mul_ne_zero hx hy)).trans ?_)
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Associated (HMul.hMul x y) (HAdd.hAdd (UniqueFactorizationMonoid.factors x)  …
  -/
  rw [Multiset.prod_add]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x y : α
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Associated (HMul.hMul x y) (HMul.hMul (UniqueFactorizationMonoid.factors x). …
  -/
  exact (Associated.mul_mul (factors_prod hx) (factors_prod hy)).symm
  /-
    🎉 no goals
  -/


theorem factors_pow {x : α} (n : ℕ) :
    Multiset.Rel Associated (factors (x ^ n)) (n • factors x) := by
  match n with
  | 0 => rw [zero_smul, pow_zero, factors_one, Multiset.rel_zero_right]
  | n+1 =>
    by_cases h0 : x = 0
    · simp [h0, zero_pow n.succ_ne_zero, smul_zero]
    · rw [pow_succ', succ_nsmul']
      refine Multiset.Rel.trans _ (factors_mul h0 (pow_ne_zero n h0)) ?_
      refine Multiset.Rel.add ?_ <| factors_pow n
      exact Multiset.rel_refl_of_refl_on fun y _ => Associated.refl _


@[simp]
theorem factors_pos (x : α) (hx : x ≠ 0) : 0 < factors x ↔ ¬IsUnit x := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    x : α
    hx : Ne x 0
    ⊢ Iff (LT.lt 0 (UniqueFactorizationMonoid.factors x)) (Not (IsUnit x))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      ⊢ LT.lt 0 (UniqueFactorizationMonoid.factors x) → Not (IsUnit x)
    -/
  · intro h hx
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx✝ : Ne x 0
      h : LT.lt 0 (UniqueFactorizationMonoid.factors x)
      hx : IsUnit x
      ⊢ False
    -/
    obtain ⟨p, hp⟩ := Multiset.exists_mem_of_ne_zero h.ne'
    /-
      case mp.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx✝ : Ne x 0
      h : LT.lt 0 (UniqueFactorizationMonoid.factors x)
      hx : IsUnit x
      p : α
      hp : Membership.mem (UniqueFactorizationMonoid.factors x) p
      ⊢ False
    -/
    exact (prime_of_factor _ hp).not_unit (isUnit_of_dvd_unit (dvd_of_mem_factors hp) hx)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      ⊢ Not (IsUnit x) → LT.lt 0 (UniqueFactorizationMonoid.factors x)
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      x : α
      hx : Ne x 0
      h : Not (IsUnit x)
      ⊢ LT.lt 0 (UniqueFactorizationMonoid.factors x)
    -/
    obtain ⟨p, hp⟩ := exists_mem_factors hx h
    exact
      bot_lt_iff_ne_bot.mpr
        (mt Multiset.eq_zero_iff_forall_not_mem.mp (not_forall.mpr ⟨p, not_not.mpr hp⟩))


open Multiset in
theorem factors_pow_count_prod [DecidableEq α] {x : α} (hx : x ≠ 0) :
    (∏ p ∈ (factors x).toFinset, p ^ (factors x).count p) ~ᵤ x :=
  calc
  _ = prod (∑ a ∈ toFinset (factors x), count a (factors x) • {a}) := by
    /-
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : DecidableEq α
      x : α
      hx : Ne x 0
      ⊢ Eq ((UniqueFactorizationMonoid.factors x).toFinset.prod fun p => HPow.hPow p …
    -/
    simp only [prod_sum, prod_nsmul, prod_singleton]
    /-
      🎉 no goals
    -/
                             /-
                               α : Type u_1
                               inst✝² : CancelCommMonoidWithZero α
                               inst✝¹ : UniqueFactorizationMonoid α
                               inst✝ : DecidableEq α
                               x : α
                               hx : Ne x 0
                               ⊢ Eq ((UniqueFactorizationMonoid.factors x).toFinset.sum fun a => HSMul.hSMul  …
                             -/
  _ = prod (factors x) := by rw [toFinset_sum_count_nsmul_eq (factors x)]
                             /-
                               🎉 no goals
                             -/
  _ ~ᵤ x := factors_prod hx


theorem factors_rel_of_associated {a b : α} (h : Associated a b) :
    Multiset.Rel Associated (factors a) (factors b) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    h : Associated a b
    ⊢ Multiset.Rel Associated (UniqueFactorizationMonoid.factors a) (UniqueFactori …
  -/
  rcases iff_iff_and_or_not_and_not.mp h.eq_zero_iff with (⟨rfl, rfl⟩ | ⟨ha, hb⟩)
    /-
      case inl.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      h : Associated 0 0
      ⊢ Multiset.Rel Associated (UniqueFactorizationMonoid.factors 0) (UniqueFactori …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      h : Associated a b
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ⊢ Multiset.Rel Associated (UniqueFactorizationMonoid.factors a) (UniqueFactori …
    -/
  · refine factors_unique irreducible_of_factor irreducible_of_factor ?_
    /-
      case inr.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      h : Associated a b
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ⊢ Associated (UniqueFactorizationMonoid.factors a).prod (UniqueFactorizationMo …
    -/
    exact ((factors_prod ha).trans h).trans (factors_prod hb).symm
    /-
      🎉 no goals
    -/


theorem unique' {p q : Multiset (Associates α)} :
    (∀ a ∈ p, Irreducible a) → (∀ a ∈ q, Irreducible a) → p.prod = q.prod → p = q := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p q : Multiset (Associates α)
    ⊢ (∀ (a : Associates α), Membership.mem p a → Irreducible a) → (∀ (a : Associa …
  -/
  apply Multiset.induction_on_multiset_quot p
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p q : Multiset (Associates α)
    ⊢ ∀ (s : Multiset α), (∀ (a : Associates α), Membership.mem (Multiset.map (Quo …
  -/
  apply Multiset.induction_on_multiset_quot q
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p q : Multiset (Associates α)
    ⊢ ∀ (s s_1 : Multiset α), (∀ (a : Associates α), Membership.mem (Multiset.map  …
  -/
  intro s t hs ht eq
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p q : Multiset (Associates α)
    s t : Multiset α
    hs : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
    ht : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
    eq : Eq (Multiset.map (Quot.mk ⇑(Associated.setoid α)) t).prod (Multiset.map ( …
    ⊢ Eq (Multiset.map (Quot.mk ⇑(Associated.setoid α)) t) (Multiset.map (Quot.mk  …
  -/
  refine Multiset.map_mk_eq_map_mk_of_rel (UniqueFactorizationMonoid.factors_unique ?_ ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      p q : Multiset (Associates α)
      s t : Multiset α
      hs : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
      ht : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
      eq : Eq (Multiset.map (Quot.mk ⇑(Associated.setoid α)) t).prod (Multiset.map ( …
      ⊢ ∀ (x : α), Membership.mem t x → Irreducible x
    -/
  · exact fun a ha => irreducible_mk.1 <| hs _ <| Multiset.mem_map_of_mem _ ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      p q : Multiset (Associates α)
      s t : Multiset α
      hs : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
      ht : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
      eq : Eq (Multiset.map (Quot.mk ⇑(Associated.setoid α)) t).prod (Multiset.map ( …
      ⊢ ∀ (x : α), Membership.mem s x → Irreducible x
    -/
  · exact fun a ha => irreducible_mk.1 <| ht _ <| Multiset.mem_map_of_mem _ ha
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p q : Multiset (Associates α)
    s t : Multiset α
    hs : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
    ht : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
    eq : Eq (Multiset.map (Quot.mk ⇑(Associated.setoid α)) t).prod (Multiset.map ( …
    ⊢ Associated t.prod s.prod
  -/
  have eq' : (Quot.mk Setoid.r : α → Associates α) = Associates.mk := funext quot_mk_eq_mk
  /-
    case refine_3
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    p q : Multiset (Associates α)
    s t : Multiset α
    hs : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
    ht : ∀ (a : Associates α), Membership.mem (Multiset.map (Quot.mk ⇑(Associated. …
    eq : Eq (Multiset.map (Quot.mk ⇑(Associated.setoid α)) t).prod (Multiset.map ( …
    eq' : Eq (Quot.mk ⇑(Associated.setoid α)) Associates.mk
    ⊢ Associated t.prod s.prod
  -/
  rwa [eq', prod_mk, prod_mk, mk_eq_mk_iff_associated] at eq
  /-
    🎉 no goals
  -/


theorem prod_le_prod_iff_le [Nontrivial α] {p q : Multiset (Associates α)}
    (hp : ∀ a ∈ p, Irreducible a) (hq : ∀ a ∈ q, Irreducible a) : p.prod ≤ q.prod ↔ p ≤ q := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    p q : Multiset (Associates α)
    hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
    hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
    ⊢ Iff (LE.le p.prod q.prod) (LE.le p q)
  -/
  refine ⟨?_, prod_le_prod⟩
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    p q : Multiset (Associates α)
    hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
    hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
    ⊢ LE.le p.prod q.prod → LE.le p q
  -/
  rintro ⟨c, eqc⟩
  /-
    case intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    p q : Multiset (Associates α)
    hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
    hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
    c : Associates α
    eqc : Eq q.prod (HMul.hMul p.prod c)
    ⊢ LE.le p q
  -/
  refine Multiset.le_iff_exists_add.2 ⟨factors c, unique' hq (fun x hx ↦ ?_) ?_⟩
    /-
      case intro.refine_1
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      p q : Multiset (Associates α)
      hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
      hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
      c : Associates α
      eqc : Eq q.prod (HMul.hMul p.prod c)
      x : Associates α
      hx : Membership.mem (HAdd.hAdd p (UniqueFactorizationMonoid.factors c)) x
      ⊢ Irreducible x
    -/
  · obtain h | h := Multiset.mem_add.1 hx
      /-
        case intro.refine_1.inl
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        inst✝ : Nontrivial α
        p q : Multiset (Associates α)
        hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
        hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
        c : Associates α
        eqc : Eq q.prod (HMul.hMul p.prod c)
        x : Associates α
        hx : Membership.mem (HAdd.hAdd p (UniqueFactorizationMonoid.factors c)) x
        h : Membership.mem p x
        ⊢ Irreducible x
      -/
    · exact hp x h
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_1.inr
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        inst✝ : Nontrivial α
        p q : Multiset (Associates α)
        hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
        hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
        c : Associates α
        eqc : Eq q.prod (HMul.hMul p.prod c)
        x : Associates α
        hx : Membership.mem (HAdd.hAdd p (UniqueFactorizationMonoid.factors c)) x
        h : Membership.mem (UniqueFactorizationMonoid.factors c) x
        ⊢ Irreducible x
      -/
    · exact irreducible_of_factor _ h
      /-
        🎉 no goals
      -/
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      p q : Multiset (Associates α)
      hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
      hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
      c : Associates α
      eqc : Eq q.prod (HMul.hMul p.prod c)
      ⊢ Eq q.prod (HAdd.hAdd p (UniqueFactorizationMonoid.factors c)).prod
    -/
  · rw [eqc, Multiset.prod_add]
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      p q : Multiset (Associates α)
      hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
      hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
      c : Associates α
      eqc : Eq q.prod (HMul.hMul p.prod c)
      ⊢ Eq (HMul.hMul p.prod c) (HMul.hMul p.prod (UniqueFactorizationMonoid.factors …
    -/
    congr
    /-
      case intro.refine_2.e_a
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      p q : Multiset (Associates α)
      hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
      hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
      c : Associates α
      eqc : Eq q.prod (HMul.hMul p.prod c)
      ⊢ Eq c (UniqueFactorizationMonoid.factors c).prod
    -/
    refine associated_iff_eq.mp (factors_prod fun hc => ?_).symm
    /-
      case intro.refine_2.e_a
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      p q : Multiset (Associates α)
      hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
      hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
      c : Associates α
      eqc : Eq q.prod (HMul.hMul p.prod c)
      hc : Eq c 0
      ⊢ False
    -/
    refine not_irreducible_zero (hq _ ?_)
    /-
      case intro.refine_2.e_a
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      p q : Multiset (Associates α)
      hp : ∀ (a : Associates α), Membership.mem p a → Irreducible a
      hq : ∀ (a : Associates α), Membership.mem q a → Irreducible a
      c : Associates α
      eqc : Eq q.prod (HMul.hMul p.prod c)
      hc : Eq c 0
      ⊢ Membership.mem q 0
    -/
    rw [← prod_eq_zero_iff, eqc, hc, mul_zero]
    /-
      🎉 no goals
    -/


theorem WfDvdMonoid.of_exists_prime_factors : WfDvdMonoid α :=
  ⟨by
    classical
      refine RelHomClass.wellFounded
        (RelHom.mk ?_ ?_ : (DvdNotUnit : α → α → Prop) →r ((· < ·) : ℕ∞ → ℕ∞ → Prop)) wellFounded_lt
      · intro a
        by_cases h : a = 0
        · exact ⊤
        exact ↑(Multiset.card (Classical.choose (pf a h)))
      rintro a b ⟨ane0, ⟨c, hc, b_eq⟩⟩
      rw [dif_neg ane0]
      by_cases h : b = 0
      · simp [h, lt_top_iff_ne_top]
      · rw [dif_neg h, Nat.cast_lt]
        have cne0 : c ≠ 0 := by
          refine mt (fun con => ?_) h
          rw [b_eq, con, mul_zero]
        calc
          Multiset.card (Classical.choose (pf a ane0)) <
              _ + Multiset.card (Classical.choose (pf c cne0)) :=
            lt_add_of_pos_right _
              (Multiset.card_pos.mpr fun con => hc (associated_one_iff_isUnit.mp ?_))
          _ = Multiset.card (Classical.choose (pf a ane0) + Classical.choose (pf c cne0)) :=
            (Multiset.card_add _ _).symm
          _ = Multiset.card (Classical.choose (pf b h)) :=
            Multiset.card_eq_card_of_rel
            (prime_factors_unique ?_ (Classical.choose_spec (pf _ h)).1 ?_)

        · convert (Classical.choose_spec (pf c cne0)).2.symm
          rw [con, Multiset.prod_zero]
        · intro x hadd
          rw [Multiset.mem_add] at hadd
          cases' hadd with h h <;> apply (Classical.choose_spec (pf _ _)).1 _ h <;> assumption
        · rw [Multiset.prod_add]
          trans a * c
          · apply Associated.mul_mul <;> apply (Classical.choose_spec (pf _ _)).2 <;> assumption
          · rw [← b_eq]
            apply (Classical.choose_spec (pf _ _)).2.symm; assumption⟩


theorem irreducible_iff_prime_of_exists_prime_factors {p : α} : Irreducible p ↔ Prime p := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    p : α
    ⊢ Iff (Irreducible p) (Prime p)
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
      p : α
      hp0 : Eq p 0
      ⊢ Iff (Irreducible p) (Prime p)
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    p : α
    hp0 : Not (Eq p 0)
    ⊢ Iff (Irreducible p) (Prime p)
  -/
  refine ⟨fun h => ?_, Prime.irreducible⟩
  /-
    case neg
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    p : α
    hp0 : Not (Eq p 0)
    h : Irreducible p
    ⊢ Prime p
  -/
  obtain ⟨f, hf⟩ := pf p hp0
  /-
    case neg.intro
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    p : α
    hp0 : Not (Eq p 0)
    h : Irreducible p
    f : Multiset α
    hf : And (∀ (b : α), Membership.mem f b → Prime b) (Associated f.prod p)
    ⊢ Prime p
  -/
  obtain ⟨q, hq, rfl⟩ := prime_factors_irreducible h hf
  /-
    case neg.intro.intro.intro
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    p : α
    hp0 : Not (Eq p 0)
    h : Irreducible p
    q : α
    hq : Associated p q
    hf : And (∀ (b : α), Membership.mem (Singleton.singleton q) b → Prime b) (Asso …
    ⊢ Prime p
  -/
  rw [hq.prime_iff]
  /-
    case neg.intro.intro.intro
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    pf : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    p : α
    hp0 : Not (Eq p 0)
    h : Irreducible p
    q : α
    hq : Associated p q
    hf : And (∀ (b : α), Membership.mem (Singleton.singleton q) b → Prime b) (Asso …
    ⊢ Prime q
  -/
  exact hf.1 q (Multiset.mem_singleton_self _)
  /-
    🎉 no goals
  -/


theorem UniqueFactorizationMonoid.of_exists_prime_factors : UniqueFactorizationMonoid α :=
  { WfDvdMonoid.of_exists_prime_factors pf with
    irreducible_iff_prime := irreducible_iff_prime_of_exists_prime_factors pf }


theorem UniqueFactorizationMonoid.iff_exists_prime_factors [CancelCommMonoidWithZero α] :
    UniqueFactorizationMonoid α ↔
      ∀ a : α, a ≠ 0 → ∃ f : Multiset α, (∀ b ∈ f, Prime b) ∧ f.prod ~ᵤ a :=
  ⟨fun h => @UniqueFactorizationMonoid.exists_prime_factors _ _ h,
    UniqueFactorizationMonoid.of_exists_prime_factors⟩


theorem MulEquiv.uniqueFactorizationMonoid (e : α ≃* β) (hα : UniqueFactorizationMonoid α) :
    UniqueFactorizationMonoid β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : CancelCommMonoidWithZero β
    e : MulEquiv α β
    hα : UniqueFactorizationMonoid α
    ⊢ UniqueFactorizationMonoid β
  -/
  rw [UniqueFactorizationMonoid.iff_exists_prime_factors] at hα ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : CancelCommMonoidWithZero β
    e : MulEquiv α β
    hα : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b →  …
    ⊢ ∀ (a : β), Ne a 0 → Exists fun f => And (∀ (b : β), Membership.mem f b → Pri …
  -/
  intro a ha
  obtain ⟨w, hp, u, h⟩ :=
    hα (e.symm a) fun h =>
      ha <| by
        convert← map_zero e
        simp [← h]
  exact
    ⟨w.map e, fun b hb =>
        let ⟨c, hc, he⟩ := Multiset.mem_map.1 hb
        he ▸ e.prime_iff.2 (hp c hc),
        Units.map e.toMonoidHom u,
      by
        rw [Multiset.prod_hom, toMonoidHom_eq_coe, Units.coe_map, MonoidHom.coe_coe, ← map_mul e, h,
          apply_symm_apply]⟩


theorem MulEquiv.uniqueFactorizationMonoid_iff (e : α ≃* β) :
    UniqueFactorizationMonoid α ↔ UniqueFactorizationMonoid β :=
  ⟨e.uniqueFactorizationMonoid, e.symm.uniqueFactorizationMonoid⟩


theorem of_existsUnique_irreducible_factors [CancelCommMonoidWithZero α]
    (eif : ∀ a : α, a ≠ 0 → ∃ f : Multiset α, (∀ b ∈ f, Irreducible b) ∧ f.prod ~ᵤ a)
    (uif :
      ∀ f g : Multiset α,
        (∀ x ∈ f, Irreducible x) →
          (∀ x ∈ g, Irreducible x) → f.prod ~ᵤ g.prod → Multiset.Rel Associated f g) :
    UniqueFactorizationMonoid α :=
  UniqueFactorizationMonoid.of_exists_prime_factors
    (by
      /-
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
        uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
        ⊢ ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → Pri …
      -/
      convert eif using 7
      /-
        case h.h'.h.e'_2.h.h.e'_1.h.h'.a
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        eif : ∀ (a : α), Ne a 0 → Exists fun f => And (∀ (b : α), Membership.mem f b → …
        uif : ∀ (f g : Multiset α), (∀ (x : α), Membership.mem f x → Irreducible x) →  …
        a✝³ : α
        a✝² : Ne a✝³ 0
        x✝ : Multiset α
        a✝¹ : α
        a✝ : Membership.mem x✝ a✝¹
        ⊢ Iff (Prime a✝¹) (Irreducible a✝¹)
      -/
      simp_rw [irreducible_iff_prime_of_existsUnique_irreducible_factors eif uif])
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-12-17")]
alias of_exists_unique_irreducible_factors := of_existsUnique_irreducible_factors


theorem isRelPrime_iff_no_prime_factors {a b : R} (ha : a ≠ 0) :
    IsRelPrime a b ↔ ∀ ⦃d⦄, d ∣ a → d ∣ b → ¬Prime d :=
  ⟨fun h _ ha hb ↦ (·.not_unit <| h ha hb), fun h ↦ WfDvdMonoid.isRelPrime_of_no_irreducible_factors
    (ha ·.1) fun _ irr ha hb ↦ h ha hb (UniqueFactorizationMonoid.irreducible_iff_prime.mp irr)⟩


/-- Euclid's lemma: if `a ∣ b * c` and `a` and `c` have no common prime factors, `a ∣ b`.
Compare `IsCoprime.dvd_of_dvd_mul_left`. -/
theorem dvd_of_dvd_mul_left_of_no_prime_factors {a b c : R} (ha : a ≠ 0)
    (h : ∀ ⦃d⦄, d ∣ a → d ∣ c → ¬Prime d) : a ∣ b * c → a ∣ b :=
  ((isRelPrime_iff_no_prime_factors ha).mpr h).dvd_of_dvd_mul_right


/-- Euclid's lemma: if `a ∣ b * c` and `a` and `b` have no common prime factors, `a ∣ c`.
Compare `IsCoprime.dvd_of_dvd_mul_right`. -/
theorem dvd_of_dvd_mul_right_of_no_prime_factors {a b c : R} (ha : a ≠ 0)
    (no_factors : ∀ {d}, d ∣ a → d ∣ b → ¬Prime d) : a ∣ b * c → a ∣ c := by
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : UniqueFactorizationMonoid R
    a b c : R
    ha : Ne a 0
    no_factors : ∀ {d : R}, Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ⊢ Dvd.dvd a (HMul.hMul b c) → Dvd.dvd a c
  -/
  simpa [mul_comm b c] using dvd_of_dvd_mul_left_of_no_prime_factors ha @no_factors
  /-
    🎉 no goals
  -/


/-- If `a ≠ 0, b` are elements of a unique factorization domain, then dividing
out their common factor `c'` gives `a'` and `b'` with no factors in common. -/
theorem exists_reduced_factors :
    ∀ a ≠ (0 : R), ∀ b,
      ∃ a' b' c', IsRelPrime a' b' ∧ c' * a' = a ∧ c' * b' = b := by
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : UniqueFactorizationMonoid R
    ⊢ ∀ (a : R), Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun  …
  -/
  intro a
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : UniqueFactorizationMonoid R
    a : R
    ⊢ Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' => And ( …
  -/
  refine induction_on_prime a ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a : R
      ⊢ Ne 0 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' => And ( …
    -/
  · intros
    /-
      case refine_1
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a : R
      a✝ : Ne 0 0
      b✝ : R
      ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a : R
      ⊢ ∀ (x : R), IsUnit x → Ne x 0 → ∀ (b : R), Exists fun a' => Exists fun b' =>  …
    -/
  · intro a a_unit _ b
    /-
      case refine_2
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a✝¹ a : R
      a_unit : IsUnit a
      a✝ : Ne a 0
      b : R
      ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
    -/
    use a, b, 1
    /-
      case h
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a✝¹ a : R
      a_unit : IsUnit a
      a✝ : Ne a 0
      b : R
      ⊢ And (IsRelPrime a b) (And (Eq (HMul.hMul 1 a) a) (Eq (HMul.hMul 1 b) b))
    -/
    constructor
      /-
        case h.left
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝¹ a : R
        a_unit : IsUnit a
        a✝ : Ne a 0
        b : R
        ⊢ IsRelPrime a b
      -/
    · intro p p_dvd_a _
      /-
        case h.left
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝² a : R
        a_unit : IsUnit a
        a✝¹ : Ne a 0
        b p : R
        p_dvd_a : Dvd.dvd p a
        a✝ : Dvd.dvd p b
        ⊢ IsUnit p
      -/
      exact isUnit_of_dvd_unit p_dvd_a a_unit
      /-
        🎉 no goals
      -/
      /-
        case h.right
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝¹ a : R
        a_unit : IsUnit a
        a✝ : Ne a 0
        b : R
        ⊢ And (Eq (HMul.hMul 1 a) a) (Eq (HMul.hMul 1 b) b)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a : R
      ⊢ ∀ (a p : R), Ne a 0 → Prime p → (Ne a 0 → ∀ (b : R), Exists fun a' => Exists …
    -/
  · intro a p a_ne_zero p_prime ih_a pa_ne_zero b
    /-
      case refine_3
      R : Type u_2
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      a✝ a p : R
      a_ne_zero : Ne a 0
      p_prime : Prime p
      ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
      pa_ne_zero : Ne (HMul.hMul p a) 0
      b : R
      ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
    -/
    by_cases h : p ∣ b
      /-
        case pos
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝ a p : R
        a_ne_zero : Ne a 0
        p_prime : Prime p
        ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
        pa_ne_zero : Ne (HMul.hMul p a) 0
        b : R
        h : Dvd.dvd p b
        ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
      -/
    · rcases h with ⟨b, rfl⟩
      /-
        case pos.intro
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝ a p : R
        a_ne_zero : Ne a 0
        p_prime : Prime p
        ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
        pa_ne_zero : Ne (HMul.hMul p a) 0
        b : R
        ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
      -/
      obtain ⟨a', b', c', no_factor, ha', hb'⟩ := ih_a a_ne_zero b
      /-
        case pos.intro.intro.intro.intro.intro.intro
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝ a p : R
        a_ne_zero : Ne a 0
        p_prime : Prime p
        ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
        pa_ne_zero : Ne (HMul.hMul p a) 0
        b a' b' c' : R
        no_factor : IsRelPrime a' b'
        ha' : Eq (HMul.hMul c' a') a
        hb' : Eq (HMul.hMul c' b') b
        ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
      -/
      refine ⟨a', b', p * c', @no_factor, ?_, ?_⟩
        /-
          case pos.intro.intro.intro.intro.intro.intro.refine_1
          R : Type u_2
          inst✝¹ : CancelCommMonoidWithZero R
          inst✝ : UniqueFactorizationMonoid R
          a✝ a p : R
          a_ne_zero : Ne a 0
          p_prime : Prime p
          ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
          pa_ne_zero : Ne (HMul.hMul p a) 0
          b a' b' c' : R
          no_factor : IsRelPrime a' b'
          ha' : Eq (HMul.hMul c' a') a
          hb' : Eq (HMul.hMul c' b') b
          ⊢ Eq (HMul.hMul (HMul.hMul p c') a') (HMul.hMul p a)
        -/
      · rw [mul_assoc, ha']
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.intro.intro.intro.intro.intro.refine_2
          R : Type u_2
          inst✝¹ : CancelCommMonoidWithZero R
          inst✝ : UniqueFactorizationMonoid R
          a✝ a p : R
          a_ne_zero : Ne a 0
          p_prime : Prime p
          ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
          pa_ne_zero : Ne (HMul.hMul p a) 0
          b a' b' c' : R
          no_factor : IsRelPrime a' b'
          ha' : Eq (HMul.hMul c' a') a
          hb' : Eq (HMul.hMul c' b') b
          ⊢ Eq (HMul.hMul (HMul.hMul p c') b') (HMul.hMul p b)
        -/
      · rw [mul_assoc, hb']
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a✝ a p : R
        a_ne_zero : Ne a 0
        p_prime : Prime p
        ih_a : Ne a 0 → ∀ (b : R), Exists fun a' => Exists fun b' => Exists fun c' =>  …
        pa_ne_zero : Ne (HMul.hMul p a) 0
        b : R
        h : Not (Dvd.dvd p b)
        ⊢ Exists fun a' => Exists fun b' => Exists fun c' => And (IsRelPrime a' b') (A …
      -/
    · obtain ⟨a', b', c', coprime, rfl, rfl⟩ := ih_a a_ne_zero b
      /-
        case neg.intro.intro.intro.intro.intro
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a p : R
        p_prime : Prime p
        a' b' c' : R
        coprime : IsRelPrime a' b'
        a_ne_zero : Ne (HMul.hMul c' a') 0
        ih_a : Ne (HMul.hMul c' a') 0 → ∀ (b : R), Exists fun a'_1 => Exists fun b' => …
        pa_ne_zero : Ne (HMul.hMul p (HMul.hMul c' a')) 0
        h : Not (Dvd.dvd p (HMul.hMul c' b'))
        ⊢ Exists fun a'_1 => Exists fun b'_1 => Exists fun c'_1 => And (IsRelPrime a'_ …
      -/
      refine ⟨p * a', b', c', ?_, mul_left_comm _ _ _, rfl⟩
      /-
        case neg.intro.intro.intro.intro.intro
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a p : R
        p_prime : Prime p
        a' b' c' : R
        coprime : IsRelPrime a' b'
        a_ne_zero : Ne (HMul.hMul c' a') 0
        ih_a : Ne (HMul.hMul c' a') 0 → ∀ (b : R), Exists fun a'_1 => Exists fun b' => …
        pa_ne_zero : Ne (HMul.hMul p (HMul.hMul c' a')) 0
        h : Not (Dvd.dvd p (HMul.hMul c' b'))
        ⊢ IsRelPrime (HMul.hMul p a') b'
      -/
      intro q q_dvd_pa' q_dvd_b'
      /-
        case neg.intro.intro.intro.intro.intro
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a p : R
        p_prime : Prime p
        a' b' c' : R
        coprime : IsRelPrime a' b'
        a_ne_zero : Ne (HMul.hMul c' a') 0
        ih_a : Ne (HMul.hMul c' a') 0 → ∀ (b : R), Exists fun a'_1 => Exists fun b' => …
        pa_ne_zero : Ne (HMul.hMul p (HMul.hMul c' a')) 0
        h : Not (Dvd.dvd p (HMul.hMul c' b'))
        q : R
        q_dvd_pa' : Dvd.dvd q (HMul.hMul p a')
        q_dvd_b' : Dvd.dvd q b'
        ⊢ IsUnit q
      -/
      cases' p_prime.left_dvd_or_dvd_right_of_dvd_mul q_dvd_pa' with p_dvd_q q_dvd_a'
        /-
          case neg.intro.intro.intro.intro.intro.inl
          R : Type u_2
          inst✝¹ : CancelCommMonoidWithZero R
          inst✝ : UniqueFactorizationMonoid R
          a p : R
          p_prime : Prime p
          a' b' c' : R
          coprime : IsRelPrime a' b'
          a_ne_zero : Ne (HMul.hMul c' a') 0
          ih_a : Ne (HMul.hMul c' a') 0 → ∀ (b : R), Exists fun a'_1 => Exists fun b' => …
          pa_ne_zero : Ne (HMul.hMul p (HMul.hMul c' a')) 0
          h : Not (Dvd.dvd p (HMul.hMul c' b'))
          q : R
          q_dvd_pa' : Dvd.dvd q (HMul.hMul p a')
          q_dvd_b' : Dvd.dvd q b'
          p_dvd_q : Dvd.dvd p q
          ⊢ IsUnit q
        -/
      · have : p ∣ c' * b' := dvd_mul_of_dvd_right (p_dvd_q.trans q_dvd_b') _
        /-
          case neg.intro.intro.intro.intro.intro.inl
          R : Type u_2
          inst✝¹ : CancelCommMonoidWithZero R
          inst✝ : UniqueFactorizationMonoid R
          a p : R
          p_prime : Prime p
          a' b' c' : R
          coprime : IsRelPrime a' b'
          a_ne_zero : Ne (HMul.hMul c' a') 0
          ih_a : Ne (HMul.hMul c' a') 0 → ∀ (b : R), Exists fun a'_1 => Exists fun b' => …
          pa_ne_zero : Ne (HMul.hMul p (HMul.hMul c' a')) 0
          h : Not (Dvd.dvd p (HMul.hMul c' b'))
          q : R
          q_dvd_pa' : Dvd.dvd q (HMul.hMul p a')
          q_dvd_b' : Dvd.dvd q b'
          p_dvd_q : Dvd.dvd p q
          this : Dvd.dvd p (HMul.hMul c' b')
          ⊢ IsUnit q
        -/
        contradiction
        /-
          🎉 no goals
        -/
      /-
        case neg.intro.intro.intro.intro.intro.inr
        R : Type u_2
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        a p : R
        p_prime : Prime p
        a' b' c' : R
        coprime : IsRelPrime a' b'
        a_ne_zero : Ne (HMul.hMul c' a') 0
        ih_a : Ne (HMul.hMul c' a') 0 → ∀ (b : R), Exists fun a'_1 => Exists fun b' => …
        pa_ne_zero : Ne (HMul.hMul p (HMul.hMul c' a')) 0
        h : Not (Dvd.dvd p (HMul.hMul c' b'))
        q : R
        q_dvd_pa' : Dvd.dvd q (HMul.hMul p a')
        q_dvd_b' : Dvd.dvd q b'
        q_dvd_a' : Dvd.dvd q a'
        ⊢ IsUnit q
      -/
      exact coprime q_dvd_a' q_dvd_b'
      /-
        🎉 no goals
      -/


theorem exists_reduced_factors' (a b : R) (hb : b ≠ 0) :
    ∃ a' b' c', IsRelPrime a' b' ∧ c' * a' = a ∧ c' * b' = b :=
  let ⟨b', a', c', no_factor, hb, ha⟩ := exists_reduced_factors b hb a
  ⟨a', b', c', fun _ hpb hpa => no_factor hpa hpb, ha, hb⟩


@[deprecated (since := "2024-09-21")] alias pow_right_injective := pow_injective_of_not_isUnit

@[deprecated (since := "2024-09-21")] alias pow_eq_pow_iff := pow_inj_of_not_isUnit


