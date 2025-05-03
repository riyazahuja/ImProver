instance (priority := 100) isScalarTower_right {α} [SMul α R] [IsScalarTower α R R] :
    IsScalarTower α (R ⧸ I) (R ⧸ I) :=
  (Quotient.ringCon I).isScalarTower_right


instance smulCommClass {α} [SMul α R] [IsScalarTower α R R] [SMulCommClass α R R] :
    SMulCommClass α (R ⧸ I) (R ⧸ I) :=
  (Quotient.ringCon I).smulCommClass


instance smulCommClass' {α} [SMul α R] [IsScalarTower α R R] [SMulCommClass R α R] :
    SMulCommClass (R ⧸ I) α (R ⧸ I) :=
  (Quotient.ringCon I).smulCommClass'


theorem eq_zero_iff_dvd (x y : R) : Ideal.Quotient.mk (Ideal.span ({x} : Set R)) y = 0 ↔ x ∣ y := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    ⊢ Iff (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton x))) y) 0) (Dvd …
  -/
  rw [Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_span_singleton]
  /-
    🎉 no goals
  -/


@[simp]
lemma mk_singleton_self (x : R) : mk (Ideal.span {x}) x = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    x : R
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton x))) x) 0
  -/
  rw [eq_zero_iff_dvd]
  /-
    🎉 no goals
  -/


theorem zero_eq_one_iff {I : Ideal R} : (0 : R ⧸ I) = 1 ↔ I = ⊤ :=
  eq_comm.trans <| eq_zero_iff_mem.trans (eq_top_iff_one _).symm


theorem zero_ne_one_iff {I : Ideal R} : (0 : R ⧸ I) ≠ 1 ↔ I ≠ ⊤ :=
  not_congr zero_eq_one_iff


protected theorem nontrivial {I : Ideal R} (hI : I ≠ ⊤) : Nontrivial (R ⧸ I) :=
  ⟨⟨0, 1, zero_ne_one_iff.2 hI⟩⟩


theorem subsingleton_iff {I : Ideal R} : Subsingleton (R ⧸ I) ↔ I = ⊤ := by
  rw [eq_top_iff_one, ← subsingleton_iff_zero_eq_one, eq_comm, ← (mk I).map_one,
    Quotient.eq_zero_iff_mem]


instance : Unique (R ⧸ (⊤ : Ideal R)) :=
           /-
             R : Type u
             inst✝ : CommRing R
             I : Ideal R
             a b : R
             S : Type v
             x y : R
             ⊢ ∀ (a : HasQuotient.Quotient R Top.top), Eq a Inhabited.default
           -/
  ⟨⟨0⟩, by rintro ⟨x⟩; exact Quotient.eq_zero_iff_mem.mpr Submodule.mem_top⟩
                       /-
                         🎉 no goals
                       -/


instance noZeroDivisors (I : Ideal R) [hI : I.IsPrime] : NoZeroDivisors (R ⧸ I) where
    eq_zero_or_eq_zero_of_mul_eq_zero {a b} := Quotient.inductionOn₂' a b fun {_ _} hab =>
      (hI.mem_or_mem (eq_zero_iff_mem.1 hab)).elim (Or.inl ∘ eq_zero_iff_mem.2)
        (Or.inr ∘ eq_zero_iff_mem.2)


instance isDomain (I : Ideal R) [hI : I.IsPrime] : IsDomain (R ⧸ I) :=
  let _ := Quotient.nontrivial hI.1
  NoZeroDivisors.to_isDomain _


theorem isDomain_iff_prime (I : Ideal R) : IsDomain (R ⧸ I) ↔ I.IsPrime := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Iff (IsDomain (HasQuotient.Quotient R I)) I.IsPrime
  -/
  refine ⟨fun H => ⟨zero_ne_one_iff.1 ?_, fun {x y} h => ?_⟩, fun h => inferInstance⟩
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : IsDomain (HasQuotient.Quotient R I)
      ⊢ Ne 0 1
    -/
  · haveI : Nontrivial (R ⧸ I) := ⟨H.2.1⟩
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : IsDomain (HasQuotient.Quotient R I)
      this : Nontrivial (HasQuotient.Quotient R I)
      ⊢ Ne 0 1
    -/
    exact zero_ne_one
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : IsDomain (HasQuotient.Quotient R I)
      x y : R
      h : Membership.mem I (HMul.hMul x y)
      ⊢ Or (Membership.mem I x) (Membership.mem I y)
    -/
  · simp only [← eq_zero_iff_mem, (mk I).map_mul] at h ⊢
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : IsDomain (HasQuotient.Quotient R I)
      x y : R
      h : Eq (HMul.hMul ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)) 0
      ⊢ Or (Eq ((Ideal.Quotient.mk I) x) 0) (Eq ((Ideal.Quotient.mk I) y) 0)
    -/
    haveI := @IsDomain.to_noZeroDivisors (R ⧸ I) _ H
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : IsDomain (HasQuotient.Quotient R I)
      x y : R
      h : Eq (HMul.hMul ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)) 0
      this : NoZeroDivisors (HasQuotient.Quotient R I)
      ⊢ Or (Eq ((Ideal.Quotient.mk I) x) 0) (Eq ((Ideal.Quotient.mk I) y) 0)
    -/
    exact eq_zero_or_eq_zero_of_mul_eq_zero h
    /-
      🎉 no goals
    -/


theorem exists_inv {I : Ideal R} [hI : I.IsMaximal] :
    ∀ {a : R ⧸ I}, a ≠ 0 → ∃ b : R ⧸ I, a * b = 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    ⊢ ∀ {a : HasQuotient.Quotient R I}, Ne a 0 → Exists fun b => Eq (HMul.hMul a b …
  -/
  rintro ⟨a⟩ h
  /-
    case mk
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    a✝ : HasQuotient.Quotient R I
    a : R
    h : Ne (Quot.mk (⇑(Submodule.quotientRel I)) a) 0
    ⊢ Exists fun b => Eq (HMul.hMul (Quot.mk (⇑(Submodule.quotientRel I)) a) b) 1
  -/
  rcases hI.exists_inv (mt eq_zero_iff_mem.2 h) with ⟨b, c, hc, abc⟩
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    a✝ : HasQuotient.Quotient R I
    a : R
    h : Ne (Quot.mk (⇑(Submodule.quotientRel I)) a) 0
    b c : R
    hc : Membership.mem I c
    abc : Eq (HAdd.hAdd (HMul.hMul b a) c) 1
    ⊢ Exists fun b => Eq (HMul.hMul (Quot.mk (⇑(Submodule.quotientRel I)) a) b) 1
  -/
  rw [mul_comm] at abc
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    a✝ : HasQuotient.Quotient R I
    a : R
    h : Ne (Quot.mk (⇑(Submodule.quotientRel I)) a) 0
    b c : R
    hc : Membership.mem I c
    abc : Eq (HAdd.hAdd (HMul.hMul a b) c) 1
    ⊢ Exists fun b => Eq (HMul.hMul (Quot.mk (⇑(Submodule.quotientRel I)) a) b) 1
  -/
  refine ⟨mk _ b, Quot.sound ?_⟩
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    a✝ : HasQuotient.Quotient R I
    a : R
    h : Ne (Quot.mk (⇑(Submodule.quotientRel I)) a) 0
    b c : R
    hc : Membership.mem I c
    abc : Eq (HAdd.hAdd (HMul.hMul a b) c) 1
    ⊢ (Ideal.Quotient.ringCon I).toSetoid ((fun x1 x2 => HMul.hMul x1 x2) a b) 1
  -/
  simp only [Submodule.quotientRel_def]
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    a✝ : HasQuotient.Quotient R I
    a : R
    h : Ne (Quot.mk (⇑(Submodule.quotientRel I)) a) 0
    b c : R
    hc : Membership.mem I c
    abc : Eq (HAdd.hAdd (HMul.hMul a b) c) 1
    ⊢ Membership.mem I (HSub.hSub (HMul.hMul a b) 1)
  -/
  rw [← eq_sub_iff_add_eq'] at abc
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsMaximal
    a✝ : HasQuotient.Quotient R I
    a : R
    h : Ne (Quot.mk (⇑(Submodule.quotientRel I)) a) 0
    b c : R
    hc : Membership.mem I c
    abc : Eq c (HSub.hSub 1 (HMul.hMul a b))
    ⊢ Membership.mem I (HSub.hSub (HMul.hMul a b) 1)
  -/
  rwa [abc, ← neg_mem_iff (G := R) (H := I), neg_sub] at hc
  /-
    🎉 no goals
  -/


open Classical in
/-- The quotient by a maximal ideal is a group with zero. This is a `def` rather than `instance`,
since users will have computable inverses in some applications.

See note [reducible non-instances]. -/
protected noncomputable abbrev groupWithZero (I : Ideal R) [hI : I.IsMaximal] :
    GroupWithZero (R ⧸ I) :=
  { inv := fun a => if ha : a = 0 then 0 else Classical.choose (exists_inv ha)
    mul_inv_cancel := fun a (ha : a ≠ 0) =>
                                 /-
                                   R : Type u
                                   inst✝ : CommRing R
                                   I✝ : Ideal R
                                   a✝ b : R
                                   S : Type v
                                   x y : R
                                   I : Ideal R
                                   hI : I.IsMaximal
                                   a : HasQuotient.Quotient R I
                                   ha : Ne a 0
                                   ⊢ Eq (HMul.hMul a (dite (Eq a 0) (fun ha => 0) fun ha => Classical.choose ⋯)) 1
                                 -/
      show a * dite _ _ _ = _ by rw [dif_neg ha]; exact Classical.choose_spec (exists_inv ha)
                                                  /-
                                                    🎉 no goals
                                                  -/
    inv_zero := dif_pos rfl }


/-- The quotient by a maximal ideal is a field. This is a `def` rather than `instance`, since users
will have computable inverses (and `qsmul`, `ratCast`) in some applications.

See note [reducible non-instances]. -/
protected noncomputable abbrev field (I : Ideal R) [I.IsMaximal] : Field (R ⧸ I) where
  __ := commRing _
  __ := Quotient.groupWithZero _
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


/-- If the quotient by an ideal is a field, then the ideal is maximal. -/
theorem maximal_of_isField (I : Ideal R) (hqf : IsField (R ⧸ I)) : I.IsMaximal := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hqf : IsField (HasQuotient.Quotient R I)
    ⊢ I.IsMaximal
  -/
  apply Ideal.isMaximal_iff.2
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hqf : IsField (HasQuotient.Quotient R I)
    ⊢ And (Not (Membership.mem I 1)) (∀ (J : Ideal R) (x : R), LE.le I J → Not (Me …
  -/
  constructor
    /-
      case left
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      ⊢ Not (Membership.mem I 1)
    -/
  · intro h
    /-
      case left
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      h : Membership.mem I 1
      ⊢ False
    -/
    rcases hqf.exists_pair_ne with ⟨⟨x⟩, ⟨y⟩, hxy⟩
    /-
      case left.intro.mk.intro.mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      h : Membership.mem I 1
      w✝¹ : HasQuotient.Quotient R I
      x : R
      w✝ : HasQuotient.Quotient R I
      y : R
      hxy : Ne (Quot.mk (⇑(Submodule.quotientRel I)) x) (Quot.mk (⇑(Submodule.quotie …
      ⊢ False
    -/
    exact hxy (Ideal.Quotient.eq.2 (mul_one (x - y) ▸ I.mul_mem_left _ h))
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      ⊢ ∀ (J : Ideal R) (x : R), LE.le I J → Not (Membership.mem I x) → Membership.m …
    -/
  · intro J x hIJ hxnI hxJ
    /-
      case right
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      J : Ideal R
      x : R
      hIJ : LE.le I J
      hxnI : Not (Membership.mem I x)
      hxJ : Membership.mem J x
      ⊢ Membership.mem J 1
    -/
    rcases hqf.mul_inv_cancel (mt Ideal.Quotient.eq_zero_iff_mem.1 hxnI) with ⟨⟨y⟩, hy⟩
    /-
      case right.intro.mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      J : Ideal R
      x : R
      hIJ : LE.le I J
      hxnI : Not (Membership.mem I x)
      hxJ : Membership.mem J x
      w✝ : HasQuotient.Quotient R I
      y : R
      hy : Eq (HMul.hMul ((Ideal.Quotient.mk I) x) (Quot.mk (⇑(Submodule.quotientRel …
      ⊢ Membership.mem J 1
    -/
    rw [← zero_add (1 : R), ← sub_self (x * y), sub_add]
    /-
      case right.intro.mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hqf : IsField (HasQuotient.Quotient R I)
      J : Ideal R
      x : R
      hIJ : LE.le I J
      hxnI : Not (Membership.mem I x)
      hxJ : Membership.mem J x
      w✝ : HasQuotient.Quotient R I
      y : R
      hy : Eq (HMul.hMul ((Ideal.Quotient.mk I) x) (Quot.mk (⇑(Submodule.quotientRel …
      ⊢ Membership.mem J (HSub.hSub (HMul.hMul x y) (HSub.hSub (HMul.hMul x y) 1))
    -/
    exact J.sub_mem (J.mul_mem_right _ hxJ) (hIJ (Ideal.Quotient.eq.1 hy))
    /-
      🎉 no goals
    -/


/-- The quotient of a ring by an ideal is a field iff the ideal is maximal. -/
theorem maximal_ideal_iff_isField_quotient (I : Ideal R) : I.IsMaximal ↔ IsField (R ⧸ I) :=
  ⟨fun h =>
    let _i := @Quotient.field _ _ I h
    Field.toIsField _,
    maximal_of_isField _⟩


/-- `R^n/I^n` is a `R/I`-module. -/
instance modulePi : Module (R ⧸ I) ((ι → R) ⧸ I.pi ι) where
  smul c m :=
    Quotient.liftOn₂' c m (fun r m => Submodule.Quotient.mk <| r • m) <| by
      /-
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        a b : R
        S ι : Type v
        c : HasQuotient.Quotient R I
        m : HasQuotient.Quotient (ι → R) (I.pi ι)
        ⊢ ∀ (a₁ : R) (a₂ : ι → R) (b₁ : R) (b₂ : ι → R), (Submodule.quotientRel I) a₁  …
      -/
      intro c₁ m₁ c₂ m₂ hc hm
      /-
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        a b : R
        S ι : Type v
        c : HasQuotient.Quotient R I
        m : HasQuotient.Quotient (ι → R) (I.pi ι)
        c₁ : R
        m₁ : ι → R
        c₂ : R
        m₂ : ι → R
        hc : (Submodule.quotientRel I) c₁ c₂
        hm : (Submodule.quotientRel (I.pi ι)) m₁ m₂
        ⊢ Eq ((fun r m => Submodule.Quotient.mk (HSMul.hSMul r m)) c₁ m₁) ((fun r m => …
      -/
      apply Ideal.Quotient.eq.2
      /-
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        a b : R
        S ι : Type v
        c : HasQuotient.Quotient R I
        m : HasQuotient.Quotient (ι → R) (I.pi ι)
        c₁ : R
        m₁ : ι → R
        c₂ : R
        m₂ : ι → R
        hc : (Submodule.quotientRel I) c₁ c₂
        hm : (Submodule.quotientRel (I.pi ι)) m₁ m₂
        ⊢ Membership.mem (I.pi ι) (HSub.hSub (HSMul.hSMul c₁ m₁) (HSMul.hSMul c₂ m₂))
      -/
      rw [Submodule.quotientRel_def] at hc hm
      /-
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        a b : R
        S ι : Type v
        c : HasQuotient.Quotient R I
        m : HasQuotient.Quotient (ι → R) (I.pi ι)
        c₁ : R
        m₁ : ι → R
        c₂ : R
        m₂ : ι → R
        hc : Membership.mem I (HSub.hSub c₁ c₂)
        hm : Membership.mem (I.pi ι) (HSub.hSub m₁ m₂)
        ⊢ Membership.mem (I.pi ι) (HSub.hSub (HSMul.hSMul c₁ m₁) (HSMul.hSMul c₂ m₂))
      -/
      intro i
      /-
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        a b : R
        S ι : Type v
        c : HasQuotient.Quotient R I
        m : HasQuotient.Quotient (ι → R) (I.pi ι)
        c₁ : R
        m₁ : ι → R
        c₂ : R
        m₂ : ι → R
        hc : Membership.mem I (HSub.hSub c₁ c₂)
        hm : Membership.mem (I.pi ι) (HSub.hSub m₁ m₂)
        i : ι
        ⊢ Membership.mem I (HSub.hSub (HSMul.hSMul c₁ m₁) (HSMul.hSMul c₂ m₂) i)
      -/
      exact I.mul_sub_mul_mem hc (hm i)
      /-
        🎉 no goals
      -/
  one_smul := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ ∀ (b : HasQuotient.Quotient (ι → R) (I.pi ι)), Eq (HSMul.hSMul 1 b) b
    -/
    rintro ⟨a⟩
    /-
      case mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b : R
      S ι : Type v
      b✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq (HSMul.hSMul 1 (Quot.mk (⇑(Submodule.quotientRel (I.pi ι))) a)) (Quot.mk  …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    /-
      case mk.convert_3
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b : R
      S ι : Type v
      b✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (HSMul.hSMul 1 a)) ((Ideal.Quotient.mk (I.p …
    -/
    congr with i; exact one_mul (a i)
                  /-
                    🎉 no goals
                  -/
  mul_smul := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ ∀ (x y : HasQuotient.Quotient R I) (b : HasQuotient.Quotient (ι → R) (I.pi ι …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩
    /-
      case mk.mk.mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b✝¹ : R
      S ι : Type v
      x✝ : HasQuotient.Quotient R I
      a : R
      y✝ : HasQuotient.Quotient R I
      b : R
      b✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      c : ι → R
      ⊢ Eq (HSMul.hSMul (HMul.hMul (Quot.mk (⇑(Submodule.quotientRel I)) a) (Quot.mk …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    /-
      case mk.mk.mk.convert_3
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b✝¹ : R
      S ι : Type v
      x✝ : HasQuotient.Quotient R I
      a : R
      y✝ : HasQuotient.Quotient R I
      b : R
      b✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      c : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (HSMul.hSMul ((fun x1 x2 => HMul.hMul x1 x2 …
    -/
    congr 1; funext i; exact mul_assoc a b (c i)
                       /-
                         🎉 no goals
                       -/
  smul_add := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ ∀ (a : HasQuotient.Quotient R I) (x y : HasQuotient.Quotient (ι → R) (I.pi ι …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩
    /-
      case mk.mk.mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝¹ b✝ : R
      S ι : Type v
      a✝ : HasQuotient.Quotient R I
      a : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      y✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      c : ι → R
      ⊢ Eq (HSMul.hSMul (Quot.mk (⇑(Submodule.quotientRel I)) a) (HAdd.hAdd (Quot.mk …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ ∀ (a : HasQuotient.Quotient R I), Eq (HSMul.hSMul a 0) 0
    -/
    /-
      case mk.mk.mk.convert_3
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝¹ b✝ : R
      S ι : Type v
      a✝ : HasQuotient.Quotient R I
      a : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      y✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      c : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (HSMul.hSMul a ((fun x1 x2 => HAdd.hAdd x1  …
    -/
    /-
      case mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝¹ b : R
      S ι : Type v
      a✝ : HasQuotient.Quotient R I
      a : R
      ⊢ Eq (HSMul.hSMul (Quot.mk (⇑(Submodule.quotientRel I)) a) 0) 0
    -/
    congr with i; exact mul_add a (b i) (c i)
    /-
      case mk.convert_3
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝¹ b : R
      S ι : Type v
      a✝ : HasQuotient.Quotient R I
      a : R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (HSMul.hSMul a 0)) ((Ideal.Quotient.mk (I.p …
    -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  smul_zero := by
    rintro ⟨a⟩
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    congr with _; exact mul_zero a
  add_smul := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ ∀ (r s : HasQuotient.Quotient R I) (x : HasQuotient.Quotient (ι → R) (I.pi ι …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩
    /-
      case mk.mk.mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b✝ : R
      S ι : Type v
      r✝ : HasQuotient.Quotient R I
      a : R
      s✝ : HasQuotient.Quotient R I
      b : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      c : ι → R
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (Quot.mk (⇑(Submodule.quotientRel I)) a) (Quot.mk …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    /-
      case mk.mk.mk.convert_3
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b✝ : R
      S ι : Type v
      r✝ : HasQuotient.Quotient R I
      a : R
      s✝ : HasQuotient.Quotient R I
      b : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      c : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (HSMul.hSMul ((fun x1 x2 => HAdd.hAdd x1 x2 …
    -/
    congr with i; exact add_mul a b (c i)
                  /-
                    🎉 no goals
                  -/
  zero_smul := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ ∀ (x : HasQuotient.Quotient (ι → R) (I.pi ι)), Eq (HSMul.hSMul 0 x) 0
    -/
    rintro ⟨a⟩
    /-
      case mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b : R
      S ι : Type v
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq (HSMul.hSMul 0 (Quot.mk (⇑(Submodule.quotientRel (I.pi ι))) a)) 0
    -/
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    /-
      case mk.convert_3
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a✝ b : R
      S ι : Type v
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (HSMul.hSMul 0 a)) ((Ideal.Quotient.mk (I.p …
    -/
    congr with i; exact zero_mul (a i)
                  /-
                    🎉 no goals
                  -/


/-- `R^n/I^n` is isomorphic to `(R/I)^n` as an `R/I`-module. -/
noncomputable def piQuotEquiv : ((ι → R) ⧸ I.pi ι) ≃ₗ[R ⧸ I] ι → (R ⧸ I) where
  toFun := fun x ↦
      Quotient.liftOn' x (fun f i => Ideal.Quotient.mk I (f i)) fun _ _ hab =>
        funext fun i => (Submodule.Quotient.eq' _).2 (QuotientAddGroup.leftRel_apply.mp hab i)
                 /-
                   R : Type u
                   inst✝ : CommRing R
                   I : Ideal R
                   a b : R
                   S ι : Type v
                   ⊢ ∀ (x y : HasQuotient.Quotient (ι → R) (I.pi ι)), Eq ((fun x => Quotient.lift …
                 -/
  map_add' := by rintro ⟨_⟩ ⟨_⟩; rfl
                                 /-
                                   🎉 no goals
                                 -/
                  /-
                    R : Type u
                    inst✝ : CommRing R
                    I : Ideal R
                    a b : R
                    S ι : Type v
                    ⊢ ∀ (m : HasQuotient.Quotient R I) (x : HasQuotient.Quotient (ι → R) (I.pi ι)) …
                  -/
  map_smul' := by rintro ⟨_⟩ ⟨_⟩; rfl
                                  /-
                                    🎉 no goals
                                  -/
  invFun := fun x ↦ Ideal.Quotient.mk (I.pi ι) fun i ↦ Quotient.out (x i)
  left_inv := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ Function.LeftInverse (fun x => (Ideal.Quotient.mk (I.pi ι)) fun i => Quotien …
    -/
    rintro ⟨x⟩
    /-
      case mk
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      x : ι → R
      ⊢ Eq ((fun x => (Ideal.Quotient.mk (I.pi ι)) fun i => Quotient.out (x i)) ({ t …
    -/
    exact Ideal.Quotient.eq.2 fun i => Ideal.Quotient.eq.1 (Quotient.out_eq' _)
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      ⊢ Function.RightInverse (fun x => (Ideal.Quotient.mk (I.pi ι)) fun i => Quotie …
    -/
    intro x
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      x : ι → HasQuotient.Quotient R I
      ⊢ Eq ({ toFun := fun x => Quotient.liftOn' x (fun f i => (Ideal.Quotient.mk I) …
    -/
    ext i
    /-
      case h
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      x : ι → HasQuotient.Quotient R I
      i : ι
      ⊢ Eq ({ toFun := fun x => Quotient.liftOn' x (fun f i => (Ideal.Quotient.mk I) …
    -/
    obtain ⟨_, _⟩ := @Quot.exists_rep _ _ (x i)
    /-
      case h.intro
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      a b : R
      S ι : Type v
      x : ι → HasQuotient.Quotient R I
      i : ι
      w✝ : R
      h✝ : Eq (Quot.mk (⇑(Submodule.quotientRel I)) w✝) (x i)
      ⊢ Eq ({ toFun := fun x => Quotient.liftOn' x (fun f i => (Ideal.Quotient.mk I) …
    -/
    convert Quotient.out_eq' (x i)
    /-
      🎉 no goals
    -/


/-- If `f : R^n → R^m` is an `R`-linear map and `I ⊆ R` is an ideal, then the image of `I^n` is
    contained in `I^m`. -/
theorem map_pi {ι : Type*} [Finite ι] {ι' : Type w} (x : ι → R) (hi : ∀ i, x i ∈ I)
    (f : (ι → R) →ₗ[R] ι' → R) (i : ι') : f x i ∈ I := by
  classical
    cases nonempty_fintype ι
    rw [pi_eq_sum_univ x]
    simp only [Finset.sum_apply, smul_eq_mul, map_sum, Pi.smul_apply, map_smul]
    exact I.sum_mem fun j _ => I.mul_mem_right _ (hi j)


open scoped Pointwise in
/-- A ring is made up of a disjoint union of cosets of an ideal. -/
lemma univ_eq_iUnion_image_add {R : Type*} [Ring R] (I : Ideal R) :
    (Set.univ (α := R)) = ⋃ x : R ⧸ I, x.out +ᵥ (I : Set R) :=
  QuotientAddGroup.univ_eq_iUnion_vadd I.toAddSubgroup


lemma _root_.Finite.of_finite_quot_finite_ideal {R : Type*} [Ring R] {I : Ideal R}
    [hI : Finite I] [h : Finite (R ⧸ I)] : Finite R :=
  @Finite.of_finite_quot_finite_addSubgroup _ _ _ hI h


