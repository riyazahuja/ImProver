/-- The image of a finitely generated ideal is finitely generated.

This is the `Ideal` version of `Submodule.FG.map`. -/
theorem FG.map {R S : Type*} [Semiring R] [Semiring S] {I : Ideal R} (h : I.FG) (f : R →+* S) :
    (I.map f).FG := by
  classical
    obtain ⟨s, hs⟩ := h
    refine ⟨s.image f, ?_⟩
    rw [Finset.coe_image, ← Ideal.map_span, hs]


theorem fg_ker_comp {R S A : Type*} [CommRing R] [CommRing S] [CommRing A] (f : R →+* S)
    (g : S →+* A) (hf : f.ker.FG) (hg : g.ker.FG) (hsur : Function.Surjective f) :
    (g.comp f).ker.FG := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  letI : Algebra R S := RingHom.toAlgebra f
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    this : Algebra R S := f.toAlgebra
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  letI : Algebra R A := RingHom.toAlgebra (g.comp f)
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    this✝ : Algebra R S := f.toAlgebra
    this : Algebra R A := (g.comp f).toAlgebra
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  letI : Algebra S A := RingHom.toAlgebra g
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    this✝¹ : Algebra R S := f.toAlgebra
    this✝ : Algebra R A := (g.comp f).toAlgebra
    this : Algebra S A := g.toAlgebra
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  letI : IsScalarTower R S A := IsScalarTower.of_algebraMap_eq fun _ => rfl
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    this✝² : Algebra R S := f.toAlgebra
    this✝¹ : Algebra R A := (g.comp f).toAlgebra
    this✝ : Algebra S A := g.toAlgebra
    this : IsScalarTower R S A := IsScalarTower.of_algebraMap_eq fun x => rfl
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  let f₁ := Algebra.linearMap R S
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    this✝² : Algebra R S := f.toAlgebra
    this✝¹ : Algebra R A := (g.comp f).toAlgebra
    this✝ : Algebra S A := g.toAlgebra
    this : IsScalarTower R S A := IsScalarTower.of_algebraMap_eq fun x => rfl
    f₁ : LinearMap (RingHom.id R) R S := Algebra.linearMap R S
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  let g₁ := (IsScalarTower.toAlgHom R S A).toLinearMap
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing A
    f : RingHom R S
    g : RingHom S A
    hf : (RingHom.ker f).FG
    hg : (RingHom.ker g).FG
    hsur : Function.Surjective ⇑f
    this✝² : Algebra R S := f.toAlgebra
    this✝¹ : Algebra R A := (g.comp f).toAlgebra
    this✝ : Algebra S A := g.toAlgebra
    this : IsScalarTower R S A := IsScalarTower.of_algebraMap_eq fun x => rfl
    f₁ : LinearMap (RingHom.id R) R S := Algebra.linearMap R S
    g₁ : LinearMap (RingHom.id R) S A := (IsScalarTower.toAlgHom R S A).toLinearMap
    ⊢ (RingHom.ker (g.comp f)).FG
  -/
  exact Submodule.fg_ker_comp f₁ g₁ hf (Submodule.fg_restrictScalars (RingHom.ker g) hg hsur) hsur
  /-
    🎉 no goals
  -/


theorem exists_radical_pow_le_of_fg {R : Type*} [CommSemiring R] (I : Ideal R) (h : I.radical.FG) :
    ∃ n : ℕ, I.radical ^ n ≤ I := by
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    I : Ideal R
    h : I.radical.FG
    ⊢ Exists fun n => LE.le (HPow.hPow I.radical n) I
  -/
  have := le_refl I.radical; revert this
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    I : Ideal R
    h : I.radical.FG
    ⊢ LE.le I.radical I.radical → Exists fun n => LE.le (HPow.hPow I.radical n) I
  -/
  refine Submodule.fg_induction _ _ (fun J => J ≤ I.radical → ∃ n : ℕ, J ^ n ≤ I) ?_ ?_ _ h
    /-
      case refine_1
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      ⊢ ∀ (x : R), (fun J => LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J  …
    -/
  · intro x hx
    /-
      case refine_1
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      x : R
      hx : LE.le (Submodule.span R (Singleton.singleton x)) I.radical
      ⊢ Exists fun n => LE.le (HPow.hPow (Submodule.span R (Singleton.singleton x))  …
    -/
    obtain ⟨n, hn⟩ := hx (subset_span (Set.mem_singleton x))
    /-
      case refine_1.intro
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      x : R
      hx : LE.le (Submodule.span R (Singleton.singleton x)) I.radical
      n : Nat
      hn : Membership.mem I (HPow.hPow x n)
      ⊢ Exists fun n => LE.le (HPow.hPow (Submodule.span R (Singleton.singleton x))  …
    -/
    exact ⟨n, by rwa [← Ideal.span, span_singleton_pow, span_le, Set.singleton_subset_iff]⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      ⊢ ∀ (M₁ M₂ : Submodule R R), (fun J => LE.le J I.radical → Exists fun n => LE. …
    -/
  · intro J K hJ hK hJK
    /-
      case refine_2
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      J K : Submodule R R
      hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
      hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
      hJK : LE.le (Max.max J K) I.radical
      ⊢ Exists fun n => LE.le (HPow.hPow (Max.max J K) n) I
    -/
    obtain ⟨n, hn⟩ := hJ fun x hx => hJK <| Ideal.mem_sup_left hx
    /-
      case refine_2.intro
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      J K : Submodule R R
      hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
      hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
      hJK : LE.le (Max.max J K) I.radical
      n : Nat
      hn : LE.le (HPow.hPow J n) I
      ⊢ Exists fun n => LE.le (HPow.hPow (Max.max J K) n) I
    -/
    obtain ⟨m, hm⟩ := hK fun x hx => hJK <| Ideal.mem_sup_right hx
    /-
      case refine_2.intro.intro
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      J K : Submodule R R
      hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
      hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
      hJK : LE.le (Max.max J K) I.radical
      n : Nat
      hn : LE.le (HPow.hPow J n) I
      m : Nat
      hm : LE.le (HPow.hPow K m) I
      ⊢ Exists fun n => LE.le (HPow.hPow (Max.max J K) n) I
    -/
    use n + m
    /-
      case h
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      J K : Submodule R R
      hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
      hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
      hJK : LE.le (Max.max J K) I.radical
      n : Nat
      hn : LE.le (HPow.hPow J n) I
      m : Nat
      hm : LE.le (HPow.hPow K m) I
      ⊢ LE.le (HPow.hPow (Max.max J K) (HAdd.hAdd n m)) I
    -/
    rw [← Ideal.add_eq_sup, add_pow, Ideal.sum_eq_sup, Finset.sup_le_iff]
    /-
      case h
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      J K : Submodule R R
      hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
      hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
      hJK : LE.le (Max.max J K) I.radical
      n : Nat
      hn : LE.le (HPow.hPow J n) I
      m : Nat
      hm : LE.le (HPow.hPow K m) I
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) b → …
    -/
    refine fun i _ => Ideal.mul_le_right.trans ?_
    /-
      case h
      R : Type u_3
      inst✝ : CommSemiring R
      I : Ideal R
      h : I.radical.FG
      J K : Submodule R R
      hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
      hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
      hJK : LE.le (Max.max J K) I.radical
      n : Nat
      hn : LE.le (HPow.hPow J n) I
      m : Nat
      hm : LE.le (HPow.hPow K m) I
      i : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
      ⊢ LE.le (HMul.hMul (HPow.hPow J i) (HPow.hPow K (HSub.hSub (HAdd.hAdd n m) i)) …
    -/
    obtain h | h := le_or_lt n i
      /-
        case h.inl
        R : Type u_3
        inst✝ : CommSemiring R
        I : Ideal R
        h✝ : I.radical.FG
        J K : Submodule R R
        hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
        hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
        hJK : LE.le (Max.max J K) I.radical
        n : Nat
        hn : LE.le (HPow.hPow J n) I
        m : Nat
        hm : LE.le (HPow.hPow K m) I
        i : Nat
        x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
        h : LE.le n i
        ⊢ LE.le (HMul.hMul (HPow.hPow J i) (HPow.hPow K (HSub.hSub (HAdd.hAdd n m) i)) …
      -/
    · apply Ideal.mul_le_right.trans ((Ideal.pow_le_pow_right h).trans hn)
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        R : Type u_3
        inst✝ : CommSemiring R
        I : Ideal R
        h✝ : I.radical.FG
        J K : Submodule R R
        hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
        hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
        hJK : LE.le (Max.max J K) I.radical
        n : Nat
        hn : LE.le (HPow.hPow J n) I
        m : Nat
        hm : LE.le (HPow.hPow K m) I
        i : Nat
        x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
        h : LT.lt i n
        ⊢ LE.le (HMul.hMul (HPow.hPow J i) (HPow.hPow K (HSub.hSub (HAdd.hAdd n m) i)) …
      -/
    · apply Ideal.mul_le_left.trans
      /-
        case h.inr
        R : Type u_3
        inst✝ : CommSemiring R
        I : Ideal R
        h✝ : I.radical.FG
        J K : Submodule R R
        hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
        hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
        hJK : LE.le (Max.max J K) I.radical
        n : Nat
        hn : LE.le (HPow.hPow J n) I
        m : Nat
        hm : LE.le (HPow.hPow K m) I
        i : Nat
        x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
        h : LT.lt i n
        ⊢ LE.le (HPow.hPow K (HSub.hSub (HAdd.hAdd n m) i)) I
      -/
      refine (Ideal.pow_le_pow_right ?_).trans hm
      /-
        case h.inr
        R : Type u_3
        inst✝ : CommSemiring R
        I : Ideal R
        h✝ : I.radical.FG
        J K : Submodule R R
        hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
        hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
        hJK : LE.le (Max.max J K) I.radical
        n : Nat
        hn : LE.le (HPow.hPow J n) I
        m : Nat
        hm : LE.le (HPow.hPow K m) I
        i : Nat
        x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
        h : LT.lt i n
        ⊢ LE.le m (HSub.hSub (HAdd.hAdd n m) i)
      -/
      rw [add_comm, Nat.add_sub_assoc h.le]
      /-
        case h.inr
        R : Type u_3
        inst✝ : CommSemiring R
        I : Ideal R
        h✝ : I.radical.FG
        J K : Submodule R R
        hJ : LE.le J I.radical → Exists fun n => LE.le (HPow.hPow J n) I
        hK : LE.le K I.radical → Exists fun n => LE.le (HPow.hPow K n) I
        hJK : LE.le (Max.max J K) I.radical
        n : Nat
        hn : LE.le (HPow.hPow J n) I
        m : Nat
        hm : LE.le (HPow.hPow K m) I
        i : Nat
        x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
        h : LT.lt i n
        ⊢ LE.le m (HAdd.hAdd m (HSub.hSub n i))
      -/
      apply Nat.le_add_right
      /-
        🎉 no goals
      -/


theorem exists_pow_le_of_le_radical_of_fg_radical {R : Type*} [CommSemiring R] {I J : Ideal R}
    (hIJ : I ≤ J.radical) (hJ : J.radical.FG) :
    ∃ k : ℕ, I ^ k ≤ J := by
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    I J : Ideal R
    hIJ : LE.le I J.radical
    hJ : J.radical.FG
    ⊢ Exists fun k => LE.le (HPow.hPow I k) J
  -/
  obtain ⟨k, hk⟩ := J.exists_radical_pow_le_of_fg hJ
  /-
    case intro
    R : Type u_3
    inst✝ : CommSemiring R
    I J : Ideal R
    hIJ : LE.le I J.radical
    hJ : J.radical.FG
    k : Nat
    hk : LE.le (HPow.hPow J.radical k) J
    ⊢ Exists fun k => LE.le (HPow.hPow I k) J
  -/
  use k
  calc
    I ^ k ≤ J.radical ^ k := Ideal.pow_right_mono hIJ _
    _ ≤ J := hk


@[deprecated (since := "2024-10-24")]
alias exists_pow_le_of_le_radical_of_fG := exists_pow_le_of_le_radical_of_fg_radical


lemma exists_pow_le_of_le_radical_of_fg {R : Type*} [CommSemiring R] {I J : Ideal R}
    (h' : I ≤ J.radical) (h : I.FG) :
    ∃ n : ℕ, I ^ n ≤ J := by
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    I J : Ideal R
    h' : LE.le I J.radical
    h : I.FG
    ⊢ Exists fun n => LE.le (HPow.hPow I n) J
  -/
  revert h'
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    I J : Ideal R
    h : I.FG
    ⊢ LE.le I J.radical → Exists fun n => LE.le (HPow.hPow I n) J
  -/
  apply Submodule.fg_induction _ _ _ _ _ I h
    /-
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      ⊢ ∀ (x : R), LE.le (Submodule.span R (Singleton.singleton x)) J.radical → Exis …
    -/
  · intro x hJ
    simp only [Ideal.submodule_span_eq, Ideal.span_le,
      Set.singleton_subset_iff, SetLike.mem_coe] at hJ
    /-
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      x : R
      hJ : Membership.mem J.radical x
      ⊢ Exists fun n => LE.le (HPow.hPow (Submodule.span R (Singleton.singleton x))  …
    -/
    obtain ⟨n, hn⟩ := hJ
    /-
      case intro
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      x : R
      n : Nat
      hn : Membership.mem J (HPow.hPow x n)
      ⊢ Exists fun n => LE.le (HPow.hPow (Submodule.span R (Singleton.singleton x))  …
    -/
    refine ⟨n, by simpa [Ideal.span_singleton_pow, Ideal.span_le]⟩
    /-
      🎉 no goals
    -/
    /-
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      ⊢ ∀ (M₁ M₂ : Submodule R R), (LE.le M₁ J.radical → Exists fun n => LE.le (HPow …
    -/
  · intros I₁ I₂ h₁ h₂ hJ
    /-
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      I₁ I₂ : Submodule R R
      h₁ : LE.le I₁ J.radical → Exists fun n => LE.le (HPow.hPow I₁ n) J
      h₂ : LE.le I₂ J.radical → Exists fun n => LE.le (HPow.hPow I₂ n) J
      hJ : LE.le (Max.max I₁ I₂) J.radical
      ⊢ Exists fun n => LE.le (HPow.hPow (Max.max I₁ I₂) n) J
    -/
    obtain ⟨n₁, hn₁⟩ := h₁ (le_sup_left.trans hJ)
    /-
      case intro
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      I₁ I₂ : Submodule R R
      h₁ : LE.le I₁ J.radical → Exists fun n => LE.le (HPow.hPow I₁ n) J
      h₂ : LE.le I₂ J.radical → Exists fun n => LE.le (HPow.hPow I₂ n) J
      hJ : LE.le (Max.max I₁ I₂) J.radical
      n₁ : Nat
      hn₁ : LE.le (HPow.hPow I₁ n₁) J
      ⊢ Exists fun n => LE.le (HPow.hPow (Max.max I₁ I₂) n) J
    -/
    obtain ⟨n₂, hn₂⟩ := h₂ (le_sup_right.trans hJ)
    /-
      case intro.intro
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      I₁ I₂ : Submodule R R
      h₁ : LE.le I₁ J.radical → Exists fun n => LE.le (HPow.hPow I₁ n) J
      h₂ : LE.le I₂ J.radical → Exists fun n => LE.le (HPow.hPow I₂ n) J
      hJ : LE.le (Max.max I₁ I₂) J.radical
      n₁ : Nat
      hn₁ : LE.le (HPow.hPow I₁ n₁) J
      n₂ : Nat
      hn₂ : LE.le (HPow.hPow I₂ n₂) J
      ⊢ Exists fun n => LE.le (HPow.hPow (Max.max I₁ I₂) n) J
    -/
    use n₁ + n₂
    /-
      case h
      R : Type u_3
      inst✝ : CommSemiring R
      I J : Ideal R
      h : I.FG
      I₁ I₂ : Submodule R R
      h₁ : LE.le I₁ J.radical → Exists fun n => LE.le (HPow.hPow I₁ n) J
      h₂ : LE.le I₂ J.radical → Exists fun n => LE.le (HPow.hPow I₂ n) J
      hJ : LE.le (Max.max I₁ I₂) J.radical
      n₁ : Nat
      hn₁ : LE.le (HPow.hPow I₁ n₁) J
      n₂ : Nat
      hn₂ : LE.le (HPow.hPow I₂ n₂) J
      ⊢ LE.le (HPow.hPow (Max.max I₁ I₂) (HAdd.hAdd n₁ n₂)) J
    -/
    exact Ideal.sup_pow_add_le_pow_sup_pow.trans (sup_le hn₁ hn₂)
    /-
      🎉 no goals
    -/


