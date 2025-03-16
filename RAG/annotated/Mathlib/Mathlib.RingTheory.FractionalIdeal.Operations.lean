theorem _root_.IsFractional.map (g : P →ₐ[R] P') {I : Submodule R P} :
    IsFractional S I → IsFractional S (Submodule.map g.toLinearMap I)
  | ⟨a, a_nonzero, hI⟩ =>
    ⟨a, a_nonzero, fun b hb => by
      /-
        R : Type u_1
        inst✝⁴ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝³ : CommRing P
        inst✝² : Algebra R P
        P' : Type u_3
        inst✝¹ : CommRing P'
        inst✝ : Algebra R P'
        g : AlgHom R P P'
        I : Submodule R P
        a : R
        a_nonzero : Membership.mem S a
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P'
        hb : Membership.mem (Submodule.map g.toLinearMap I) b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
      -/
      obtain ⟨b', b'_mem, hb'⟩ := Submodule.mem_map.mp hb
      /-
        case intro.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝³ : CommRing P
        inst✝² : Algebra R P
        P' : Type u_3
        inst✝¹ : CommRing P'
        inst✝ : Algebra R P'
        g : AlgHom R P P'
        I : Submodule R P
        a : R
        a_nonzero : Membership.mem S a
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P'
        hb : Membership.mem (Submodule.map g.toLinearMap I) b
        b' : P
        b'_mem : Membership.mem I b'
        hb' : Eq (g.toLinearMap b') b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
      -/
      rw [AlgHom.toLinearMap_apply] at hb'
      /-
        case intro.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝³ : CommRing P
        inst✝² : Algebra R P
        P' : Type u_3
        inst✝¹ : CommRing P'
        inst✝ : Algebra R P'
        g : AlgHom R P P'
        I : Submodule R P
        a : R
        a_nonzero : Membership.mem S a
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P'
        hb : Membership.mem (Submodule.map g.toLinearMap I) b
        b' : P
        b'_mem : Membership.mem I b'
        hb' : Eq (g b') b
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
      -/
      obtain ⟨x, hx⟩ := hI b' b'_mem
      /-
        case intro.intro.intro
        R : Type u_1
        inst✝⁴ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝³ : CommRing P
        inst✝² : Algebra R P
        P' : Type u_3
        inst✝¹ : CommRing P'
        inst✝ : Algebra R P'
        g : AlgHom R P P'
        I : Submodule R P
        a : R
        a_nonzero : Membership.mem S a
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P'
        hb : Membership.mem (Submodule.map g.toLinearMap I) b
        b' : P
        b'_mem : Membership.mem I b'
        hb' : Eq (g b') b
        x : R
        hx : Eq ((algebraMap R P) x) (HSMul.hSMul a b')
        ⊢ IsLocalization.IsInteger R (HSMul.hSMul a b)
      -/
      use x
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝³ : CommRing P
        inst✝² : Algebra R P
        P' : Type u_3
        inst✝¹ : CommRing P'
        inst✝ : Algebra R P'
        g : AlgHom R P P'
        I : Submodule R P
        a : R
        a_nonzero : Membership.mem S a
        hI : ∀ (b : P), Membership.mem I b → IsLocalization.IsInteger R (HSMul.hSMul a …
        b : P'
        hb : Membership.mem (Submodule.map g.toLinearMap I) b
        b' : P
        b'_mem : Membership.mem I b'
        hb' : Eq (g b') b
        x : R
        hx : Eq ((algebraMap R P) x) (HSMul.hSMul a b')
        ⊢ Eq ((algebraMap R P') x) (HSMul.hSMul a b)
      -/
      rw [← g.commutes, hx, _root_.map_smul, hb']⟩
      /-
        🎉 no goals
      -/


/-- `I.map g` is the pushforward of the fractional ideal `I` along the algebra morphism `g` -/
def map (g : P →ₐ[R] P') : FractionalIdeal S P → FractionalIdeal S P' := fun I =>
  ⟨Submodule.map g.toLinearMap I, I.isFractional.map g⟩


@[simp, norm_cast]
theorem coe_map (g : P →ₐ[R] P') (I : FractionalIdeal S P) :
    ↑(map g I) = Submodule.map g.toLinearMap I :=
  rfl


@[simp]
theorem mem_map {I : FractionalIdeal S P} {g : P →ₐ[R] P'} {y : P'} :
    y ∈ I.map g ↔ ∃ x, x ∈ I ∧ g x = y :=
  Submodule.mem_map


@[simp]
theorem map_id : I.map (AlgHom.id _ _) = I :=
  coeToSubmodule_injective (Submodule.map_id (I : Submodule R P))


@[simp]
theorem map_comp (g' : P' →ₐ[R] P'') : I.map (g'.comp g) = (I.map g).map g' :=
  coeToSubmodule_injective (Submodule.map_comp g.toLinearMap g'.toLinearMap I)


@[simp, norm_cast]
theorem map_coeIdeal (I : Ideal R) : (I : FractionalIdeal S P).map g = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    g : AlgHom R P P'
    I : Ideal R
    ⊢ Eq (FractionalIdeal.map g ↑I) ↑I
  -/
  ext x
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    g : AlgHom R P P'
    I : Ideal R
    x : P'
    ⊢ Iff (Membership.mem (FractionalIdeal.map g ↑I) x) (Membership.mem (↑I) x)
  -/
  simp only [mem_coeIdeal]
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    g : AlgHom R P P'
    I : Ideal R
    x : P'
    ⊢ Iff (Membership.mem (FractionalIdeal.map g ↑I) x) (Exists fun x' => And (Mem …
  -/
  constructor
    /-
      case a.mp
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      P' : Type u_3
      inst✝¹ : CommRing P'
      inst✝ : Algebra R P'
      g : AlgHom R P P'
      I : Ideal R
      x : P'
      ⊢ Membership.mem (FractionalIdeal.map g ↑I) x → Exists fun x' => And (Membersh …
    -/
  · rintro ⟨_, ⟨y, hy, rfl⟩, rfl⟩
    /-
      case a.mp.intro.intro.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      P' : Type u_3
      inst✝¹ : CommRing P'
      inst✝ : Algebra R P'
      g : AlgHom R P P'
      I : Ideal R
      y : R
      hy : Membership.mem (↑I) y
      ⊢ Exists fun x' => And (Membership.mem I x') (Eq ((algebraMap R P') x') (g.toL …
    -/
    exact ⟨y, hy, (g.commutes y).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case a.mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      P' : Type u_3
      inst✝¹ : CommRing P'
      inst✝ : Algebra R P'
      g : AlgHom R P P'
      I : Ideal R
      x : P'
      ⊢ (Exists fun x' => And (Membership.mem I x') (Eq ((algebraMap R P') x') x)) → …
    -/
  · rintro ⟨y, hy, rfl⟩
    /-
      case a.mpr.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝³ : CommRing P
      inst✝² : Algebra R P
      P' : Type u_3
      inst✝¹ : CommRing P'
      inst✝ : Algebra R P'
      g : AlgHom R P P'
      I : Ideal R
      y : R
      hy : Membership.mem I y
      ⊢ Membership.mem (FractionalIdeal.map g ↑I) ((algebraMap R P') y)
    -/
    exact ⟨_, ⟨y, hy, rfl⟩, g.commutes y⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_one : (1 : FractionalIdeal S P).map g = 1 :=
  map_coeIdeal g ⊤


@[simp]
theorem map_zero : (0 : FractionalIdeal S P).map g = 0 :=
  map_coeIdeal g 0


@[simp]
theorem map_add : (I + J).map g = I.map g + J.map g :=
  coeToSubmodule_injective (Submodule.map_sup _ _ _)


@[simp]
theorem map_mul : (I * J).map g = I.map g * J.map g := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    I J : FractionalIdeal S P
    g : AlgHom R P P'
    ⊢ Eq (FractionalIdeal.map g (HMul.hMul I J)) (HMul.hMul (FractionalIdeal.map g …
  -/
  simp only [mul_def]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    I J : FractionalIdeal S P
    g : AlgHom R P P'
    ⊢ Eq (FractionalIdeal.map g ⟨HMul.hMul ↑I ↑J, ⋯⟩) ⟨HMul.hMul ↑(FractionalIdeal …
  -/
  exact coeToSubmodule_injective (Submodule.map_mul _ _ _)
  /-
    🎉 no goals
  -/


@[simp]
theorem map_map_symm (g : P ≃ₐ[R] P') : (I.map (g : P →ₐ[R] P')).map (g.symm : P' →ₐ[R] P) = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    I : FractionalIdeal S P
    g : AlgEquiv R P P'
    ⊢ Eq (FractionalIdeal.map (↑g.symm) (FractionalIdeal.map (↑g) I)) I
  -/
  rw [← map_comp, g.symm_comp, map_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_symm_map (I : FractionalIdeal S P') (g : P ≃ₐ[R] P') :
    (I.map (g.symm : P' →ₐ[R] P)).map (g : P →ₐ[R] P') = I := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    P' : Type u_3
    inst✝¹ : CommRing P'
    inst✝ : Algebra R P'
    I : FractionalIdeal S P'
    g : AlgEquiv R P P'
    ⊢ Eq (FractionalIdeal.map (↑g) (FractionalIdeal.map (↑g.symm) I)) I
  -/
  rw [← map_comp, g.comp_symm, map_id]
  /-
    🎉 no goals
  -/


theorem map_mem_map {f : P →ₐ[R] P'} (h : Function.Injective f) {x : P} {I : FractionalIdeal S P} :
    f x ∈ map f I ↔ x ∈ I :=
  mem_map.trans ⟨fun ⟨_, hx', x'_eq⟩ => h x'_eq ▸ hx', fun h => ⟨x, h, rfl⟩⟩


theorem map_injective (f : P →ₐ[R] P') (h : Function.Injective f) :
    Function.Injective (map f : FractionalIdeal S P → FractionalIdeal S P') := fun _ _ hIJ =>
  ext fun _ => (map_mem_map h).symm.trans (hIJ.symm ▸ map_mem_map h)


/-- If `g` is an equivalence, `map g` is an isomorphism -/
def mapEquiv (g : P ≃ₐ[R] P') : FractionalIdeal S P ≃+* FractionalIdeal S P' where
  toFun := map g
  invFun := map g.symm
  map_add' I J := map_add I J _
  map_mul' I J := map_mul I J _
                   /-
                     R : Type u_1
                     inst✝⁶ : CommRing R
                     S : Submonoid R
                     P : Type u_2
                     inst✝⁵ : CommRing P
                     inst✝⁴ : Algebra R P
                     P' : Type u_3
                     inst✝³ : CommRing P'
                     inst✝² : Algebra R P'
                     P'' : Type u_4
                     inst✝¹ : CommRing P''
                     inst✝ : Algebra R P''
                     I✝ J : FractionalIdeal S P
                     g✝ : AlgHom R P P'
                     g : AlgEquiv R P P'
                     I : FractionalIdeal S P
                     ⊢ Eq (FractionalIdeal.map (↑g.symm) (FractionalIdeal.map (↑g) I)) I
                   -/
  left_inv I := by rw [← map_comp, AlgEquiv.symm_comp, map_id]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u_1
                      inst✝⁶ : CommRing R
                      S : Submonoid R
                      P : Type u_2
                      inst✝⁵ : CommRing P
                      inst✝⁴ : Algebra R P
                      P' : Type u_3
                      inst✝³ : CommRing P'
                      inst✝² : Algebra R P'
                      P'' : Type u_4
                      inst✝¹ : CommRing P''
                      inst✝ : Algebra R P''
                      I✝ J : FractionalIdeal S P
                      g✝ : AlgHom R P P'
                      g : AlgEquiv R P P'
                      I : FractionalIdeal S P'
                      ⊢ Eq (FractionalIdeal.map (↑g) (FractionalIdeal.map (↑g.symm) I)) I
                    -/
  right_inv I := by rw [← map_comp, AlgEquiv.comp_symm, map_id]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem coeFun_mapEquiv (g : P ≃ₐ[R] P') :
    (mapEquiv g : FractionalIdeal S P → FractionalIdeal S P') = map g :=
  rfl


@[simp]
theorem mapEquiv_apply (g : P ≃ₐ[R] P') (I : FractionalIdeal S P) : mapEquiv g I = map (↑g) I :=
  rfl


@[simp]
theorem mapEquiv_symm (g : P ≃ₐ[R] P') :
    ((mapEquiv g).symm : FractionalIdeal S P' ≃+* _) = mapEquiv g.symm :=
  rfl


@[simp]
theorem mapEquiv_refl : mapEquiv AlgEquiv.refl = RingEquiv.refl (FractionalIdeal S P) :=
                            /-
                              R : Type u_1
                              inst✝² : CommRing R
                              S : Submonoid R
                              P : Type u_2
                              inst✝¹ : CommRing P
                              inst✝ : Algebra R P
                              x : FractionalIdeal S P
                              ⊢ Eq ((FractionalIdeal.mapEquiv AlgEquiv.refl) x) ((RingEquiv.refl (Fractional …
                            -/
  RingEquiv.ext fun x => by simp
                            /-
                              🎉 no goals
                            -/


theorem isFractional_span_iff {s : Set P} :
    IsFractional S (span R s) ↔ ∃ a ∈ S, ∀ b : P, b ∈ s → IsInteger R (a • b) :=
  ⟨fun ⟨a, a_mem, h⟩ => ⟨a, a_mem, fun b hb => h b (subset_span hb)⟩, fun ⟨a, a_mem, h⟩ =>
    ⟨a, a_mem, fun _ hb =>
      span_induction (hx := hb) h
        (by
          /-
            R : Type u_1
            inst✝² : CommRing R
            S : Submonoid R
            P : Type u_2
            inst✝¹ : CommRing P
            inst✝ : Algebra R P
            s : Set P
            x✝¹ : Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem s b  …
            a : R
            a_mem : Membership.mem S a
            h : ∀ (b : P), Membership.mem s b → IsLocalization.IsInteger R (HSMul.hSMul a b)
            x✝ : P
            hb : Membership.mem (Submodule.span R s) x✝
            ⊢ IsLocalization.IsInteger R (HSMul.hSMul a 0)
          -/
          rw [smul_zero]
          /-
            R : Type u_1
            inst✝² : CommRing R
            S : Submonoid R
            P : Type u_2
            inst✝¹ : CommRing P
            inst✝ : Algebra R P
            s : Set P
            x✝¹ : Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem s b  …
            a : R
            a_mem : Membership.mem S a
            h : ∀ (b : P), Membership.mem s b → IsLocalization.IsInteger R (HSMul.hSMul a b)
            x✝ : P
            hb : Membership.mem (Submodule.span R s) x✝
            ⊢ IsLocalization.IsInteger R 0
          -/
          exact isInteger_zero)
          /-
            🎉 no goals
          -/
        (fun x y _ _ hx hy => by
          /-
            R : Type u_1
            inst✝² : CommRing R
            S : Submonoid R
            P : Type u_2
            inst✝¹ : CommRing P
            inst✝ : Algebra R P
            s : Set P
            x✝³ : Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem s b  …
            a : R
            a_mem : Membership.mem S a
            h : ∀ (b : P), Membership.mem s b → IsLocalization.IsInteger R (HSMul.hSMul a b)
            x✝² : P
            hb : Membership.mem (Submodule.span R s) x✝²
            x y : P
            x✝¹ : Membership.mem (Submodule.span R s) x
            x✝ : Membership.mem (Submodule.span R s) y
            hx : IsLocalization.IsInteger R (HSMul.hSMul a x)
            hy : IsLocalization.IsInteger R (HSMul.hSMul a y)
            ⊢ IsLocalization.IsInteger R (HSMul.hSMul a (HAdd.hAdd x y))
          -/
          rw [smul_add]
          /-
            R : Type u_1
            inst✝² : CommRing R
            S : Submonoid R
            P : Type u_2
            inst✝¹ : CommRing P
            inst✝ : Algebra R P
            s : Set P
            x✝³ : Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem s b  …
            a : R
            a_mem : Membership.mem S a
            h : ∀ (b : P), Membership.mem s b → IsLocalization.IsInteger R (HSMul.hSMul a b)
            x✝² : P
            hb : Membership.mem (Submodule.span R s) x✝²
            x y : P
            x✝¹ : Membership.mem (Submodule.span R s) x
            x✝ : Membership.mem (Submodule.span R s) y
            hx : IsLocalization.IsInteger R (HSMul.hSMul a x)
            hy : IsLocalization.IsInteger R (HSMul.hSMul a y)
            ⊢ IsLocalization.IsInteger R (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul a y))
          -/
          exact isInteger_add hx hy)
          /-
            🎉 no goals
          -/
        fun s x _ hx => by
        /-
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          s✝ : Set P
          x✝² : Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem s✝ b …
          a : R
          a_mem : Membership.mem S a
          h : ∀ (b : P), Membership.mem s✝ b → IsLocalization.IsInteger R (HSMul.hSMul a …
          x✝¹ : P
          hb : Membership.mem (Submodule.span R s✝) x✝¹
          s : R
          x : P
          x✝ : Membership.mem (Submodule.span R s✝) x
          hx : IsLocalization.IsInteger R (HSMul.hSMul a x)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul a (HSMul.hSMul s x))
        -/
        rw [smul_comm]
        /-
          R : Type u_1
          inst✝² : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝¹ : CommRing P
          inst✝ : Algebra R P
          s✝ : Set P
          x✝² : Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem s✝ b …
          a : R
          a_mem : Membership.mem S a
          h : ∀ (b : P), Membership.mem s✝ b → IsLocalization.IsInteger R (HSMul.hSMul a …
          x✝¹ : P
          hb : Membership.mem (Submodule.span R s✝) x✝¹
          s : R
          x : P
          x✝ : Membership.mem (Submodule.span R s✝) x
          hx : IsLocalization.IsInteger R (HSMul.hSMul a x)
          ⊢ IsLocalization.IsInteger R (HSMul.hSMul s (HSMul.hSMul a x))
        -/
        exact isInteger_smul hx⟩⟩
        /-
          🎉 no goals
        -/


theorem isFractional_of_fg [IsLocalization S P] {I : Submodule R P} (hI : I.FG) :
    IsFractional S I := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : Submodule R P
    hI : I.FG
    ⊢ IsFractional S I
  -/
  rcases hI with ⟨I, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : Finset P
    ⊢ IsFractional S (Submodule.span R ↑I)
  -/
  rcases exist_integer_multiples_of_finset S I with ⟨⟨s, hs1⟩, hs⟩
  /-
    case intro.intro.mk
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : Finset P
    s : R
    hs1 : Membership.mem S s
    hs : ∀ (a : P), Membership.mem I a → IsLocalization.IsInteger R (HSMul.hSMul ( …
    ⊢ IsFractional S (Submodule.span R ↑I)
  -/
  rw [isFractional_span_iff]
  /-
    case intro.intro.mk
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : Finset P
    s : R
    hs1 : Membership.mem S s
    hs : ∀ (a : P), Membership.mem I a → IsLocalization.IsInteger R (HSMul.hSMul ( …
    ⊢ Exists fun a => And (Membership.mem S a) (∀ (b : P), Membership.mem (↑I) b → …
  -/
  exact ⟨s, hs1, hs⟩
  /-
    🎉 no goals
  -/


theorem mem_span_mul_finite_of_mem_mul {I J : FractionalIdeal S P} {x : P} (hx : x ∈ I * J) :
    ∃ T T' : Finset P, (T : Set P) ⊆ I ∧ (T' : Set P) ⊆ J ∧ x ∈ span R (T * T' : Set P) :=
                                               /-
                                                 R : Type u_1
                                                 inst✝² : CommRing R
                                                 S : Submonoid R
                                                 P : Type u_2
                                                 inst✝¹ : CommRing P
                                                 inst✝ : Algebra R P
                                                 I J : FractionalIdeal S P
                                                 x : P
                                                 hx : Membership.mem (HMul.hMul I J) x
                                                 ⊢ Membership.mem (HMul.hMul ↑I ↑J) x
                                               -/
  Submodule.mem_span_mul_finite_of_mem_mul (by simpa using mem_coe.mpr hx)
                                               /-
                                                 🎉 no goals
                                               -/


theorem coeIdeal_fg (inj : Function.Injective (algebraMap R P)) (I : Ideal R) :
    FG ((I : FractionalIdeal S P) : Submodule R P) ↔ I.FG :=
  coeSubmodule_fg _ inj _


theorem fg_unit (I : (FractionalIdeal S P)ˣ) : FG (I : Submodule R P) :=
  Submodule.fg_unit <| Units.map (coeSubmoduleHom S P).toMonoidHom I


theorem fg_of_isUnit (I : FractionalIdeal S P) (h : IsUnit I) : FG (I : Submodule R P) :=
  fg_unit h.unit


theorem _root_.Ideal.fg_of_isUnit (inj : Function.Injective (algebraMap R P)) (I : Ideal R)
    (h : IsUnit (I : FractionalIdeal S P)) : I.FG := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    inj : Function.Injective ⇑(algebraMap R P)
    I : Ideal R
    h : IsUnit ↑I
    ⊢ I.FG
  -/
  rw [← coeIdeal_fg S inj I]
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝¹ : CommRing P
    inst✝ : Algebra R P
    inj : Function.Injective ⇑(algebraMap R P)
    I : Ideal R
    h : IsUnit ↑I
    ⊢ (↑↑I).FG
  -/
  exact FractionalIdeal.fg_of_isUnit (R := R) I h
  /-
    🎉 no goals
  -/


/-- `canonicalEquiv f f'` is the canonical equivalence between the fractional
ideals in `P` and in `P'`, which are both localizations of `R` at `S`. -/
noncomputable irreducible_def canonicalEquiv : FractionalIdeal S P ≃+* FractionalIdeal S P' :=
  mapEquiv
    { ringEquivOfRingEquiv P P' (RingEquiv.refl R)
                             /-
                               R : Type u_1
                               inst✝⁸ : CommRing R
                               S : Submonoid R
                               P : Type u_2
                               inst✝⁷ : CommRing P
                               inst✝⁶ : Algebra R P
                               P' : Type u_3
                               inst✝⁵ : CommRing P'
                               inst✝⁴ : Algebra R P'
                               P'' : Type u_4
                               inst✝³ : CommRing P''
                               inst✝² : Algebra R P''
                               I J : FractionalIdeal S P
                               g : AlgHom R P P'
                               inst✝¹ : IsLocalization S P
                               inst✝ : IsLocalization S P'
                               ⊢ Eq (Submonoid.map (RingEquiv.refl R).toMonoidHom S) S
                             -/
        (show S.map _ = S by rw [RingEquiv.toMonoidHom_refl, Submonoid.map_id]) with
                             /-
                               🎉 no goals
                             -/
      commutes' := fun _ => ringEquivOfRingEquiv_eq _ _ }


@[simp]
theorem mem_canonicalEquiv_apply {I : FractionalIdeal S P} {x : P'} :
    x ∈ canonicalEquiv S P P' I ↔
      ∃ y ∈ I,
        IsLocalization.map P' (RingHom.id R) (fun y (hy : y ∈ S) => show RingHom.id R y ∈ S from hy)
            (y : P) =
          x := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    P' : Type u_3
    inst✝³ : CommRing P'
    inst✝² : Algebra R P'
    inst✝¹ : IsLocalization S P
    inst✝ : IsLocalization S P'
    I : FractionalIdeal S P
    x : P'
    ⊢ Iff (Membership.mem ((FractionalIdeal.canonicalEquiv S P P') I) x) (Exists f …
  -/
  rw [canonicalEquiv, mapEquiv_apply, mem_map]
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    P' : Type u_3
    inst✝³ : CommRing P'
    inst✝² : Algebra R P'
    inst✝¹ : IsLocalization S P
    inst✝ : IsLocalization S P'
    I : FractionalIdeal S P
    x : P'
    ⊢ Iff
        (Exists fun x_1 =>
          And (Membership.mem I x_1)
            (Eq
              (↑(let __src := IsLocalization.ringEquivOfRingEquiv P P' (RingEquiv. …
                  { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commut …
                x_1)
              x))
        (Exists fun y => And (Membership.mem I y) (Eq ((IsLocalization.map P' (Rin …
  -/
  exact ⟨fun ⟨y, mem, Eq⟩ => ⟨y, mem, Eq⟩, fun ⟨y, mem, Eq⟩ => ⟨y, mem, Eq⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem canonicalEquiv_symm : (canonicalEquiv S P P').symm = canonicalEquiv S P' P :=
  RingEquiv.ext fun I =>
    SetLike.ext_iff.mpr fun x => by
      rw [mem_canonicalEquiv_apply, canonicalEquiv, mapEquiv_symm, mapEquiv_apply,
        mem_map]
      /-
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        P' : Type u_3
        inst✝³ : CommRing P'
        inst✝² : Algebra R P'
        inst✝¹ : IsLocalization S P
        inst✝ : IsLocalization S P'
        I : FractionalIdeal S P'
        x : P
        ⊢ Iff
            (Exists fun x_1 =>
              And (Membership.mem I x_1)
                (Eq
                  (↑(let __src := IsLocalization.ringEquivOfRingEquiv P P' (RingEquiv. …
                        { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, comm …
                    x_1)
                  x))
            (Exists fun y => And (Membership.mem I y) (Eq ((IsLocalization.map P (Ring …
      -/
      exact ⟨fun ⟨y, mem, Eq⟩ => ⟨y, mem, Eq⟩, fun ⟨y, mem, Eq⟩ => ⟨y, mem, Eq⟩⟩
      /-
        🎉 no goals
      -/


theorem canonicalEquiv_flip (I) : canonicalEquiv S P P' (canonicalEquiv S P' P I) = I := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    P' : Type u_3
    inst✝³ : CommRing P'
    inst✝² : Algebra R P'
    inst✝¹ : IsLocalization S P
    inst✝ : IsLocalization S P'
    I : FractionalIdeal S P'
    ⊢ Eq ((FractionalIdeal.canonicalEquiv S P P') ((FractionalIdeal.canonicalEquiv …
  -/
  rw [← canonicalEquiv_symm, RingEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem canonicalEquiv_canonicalEquiv (P'' : Type*) [CommRing P''] [Algebra R P'']
    [IsLocalization S P''] (I : FractionalIdeal S P) :
    canonicalEquiv S P' P'' (canonicalEquiv S P P' I) = canonicalEquiv S P P'' I := by
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁸ : CommRing P
    inst✝⁷ : Algebra R P
    P' : Type u_3
    inst✝⁶ : CommRing P'
    inst✝⁵ : Algebra R P'
    inst✝⁴ : IsLocalization S P
    inst✝³ : IsLocalization S P'
    P'' : Type u_5
    inst✝² : CommRing P''
    inst✝¹ : Algebra R P''
    inst✝ : IsLocalization S P''
    I : FractionalIdeal S P
    ⊢ Eq ((FractionalIdeal.canonicalEquiv S P' P'') ((FractionalIdeal.canonicalEqu …
  -/
  ext
  simp only [IsLocalization.map_map, RingHomInvPair.comp_eq₂, mem_canonicalEquiv_apply,
    exists_prop, exists_exists_and_eq_and]


theorem canonicalEquiv_trans_canonicalEquiv (P'' : Type*) [CommRing P''] [Algebra R P'']
    [IsLocalization S P''] :
    (canonicalEquiv S P P').trans (canonicalEquiv S P' P'') = canonicalEquiv S P P'' :=
  RingEquiv.ext (canonicalEquiv_canonicalEquiv S P P' P'')


@[simp]
theorem canonicalEquiv_coeIdeal (I : Ideal R) : canonicalEquiv S P P' I = I := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    P' : Type u_3
    inst✝³ : CommRing P'
    inst✝² : Algebra R P'
    inst✝¹ : IsLocalization S P
    inst✝ : IsLocalization S P'
    I : Ideal R
    ⊢ Eq ((FractionalIdeal.canonicalEquiv S P P') ↑I) ↑I
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    P' : Type u_3
    inst✝³ : CommRing P'
    inst✝² : Algebra R P'
    inst✝¹ : IsLocalization S P
    inst✝ : IsLocalization S P'
    I : Ideal R
    x✝ : P'
    ⊢ Iff (Membership.mem ((FractionalIdeal.canonicalEquiv S P P') ↑I) x✝) (Member …
  -/
  simp [IsLocalization.map_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem canonicalEquiv_self : canonicalEquiv S P P = RingEquiv.refl _ := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    ⊢ Eq (FractionalIdeal.canonicalEquiv S P P) (RingEquiv.refl (FractionalIdeal S …
  -/
  rw [← canonicalEquiv_trans_canonicalEquiv S P P]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    ⊢ Eq ((FractionalIdeal.canonicalEquiv S P P).trans (FractionalIdeal.canonicalE …
  -/
  convert (canonicalEquiv S P P).symm_trans_self
  /-
    case h.e'_2.h.e'_10
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    ⊢ Eq (FractionalIdeal.canonicalEquiv S P P) (FractionalIdeal.canonicalEquiv S  …
  -/
  exact (canonicalEquiv_symm S P P).symm
  /-
    🎉 no goals
  -/


/-- Nonzero fractional ideals contain a nonzero integer. -/
theorem exists_ne_zero_mem_isInteger [Nontrivial R] (hI : I ≠ 0) :
    ∃ x, x ≠ 0 ∧ algebraMap R K x ∈ I := by
  obtain ⟨y : K, y_mem, y_not_mem⟩ :=
    SetLike.exists_of_lt (by simpa only using bot_lt_iff_ne_bot.mpr hI)
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    inst✝ : Nontrivial R
    hI : Ne I 0
    y : K
    y_mem : Membership.mem I y
    y_not_mem : Not (Membership.mem Bot.bot y)
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem I ((algebraMap R K) x))
  -/
  have y_ne_zero : y ≠ 0 := by simpa using y_not_mem
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    inst✝ : Nontrivial R
    hI : Ne I 0
    y : K
    y_mem : Membership.mem I y
    y_not_mem : Not (Membership.mem Bot.bot y)
    y_ne_zero : Ne y 0
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem I ((algebraMap R K) x))
  -/
  obtain ⟨z, ⟨x, hx⟩⟩ := exists_integer_multiple R⁰ y
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_3
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    inst✝ : Nontrivial R
    hI : Ne I 0
    y : K
    y_mem : Membership.mem I y
    y_not_mem : Not (Membership.mem Bot.bot y)
    y_ne_zero : Ne y 0
    z : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    x : R
    hx : Eq ((algebraMap R K) x) (HSMul.hSMul (↑z) y)
    ⊢ Exists fun x => And (Ne x 0) (Membership.mem I ((algebraMap R K) x))
  -/
  refine ⟨x, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_3
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      I : FractionalIdeal (nonZeroDivisors R) K
      inst✝ : Nontrivial R
      hI : Ne I 0
      y : K
      y_mem : Membership.mem I y
      y_not_mem : Not (Membership.mem Bot.bot y)
      y_ne_zero : Ne y 0
      z : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      x : R
      hx : Eq ((algebraMap R K) x) (HSMul.hSMul (↑z) y)
      ⊢ Ne x 0
    -/
  · rw [Ne, ← @IsFractionRing.to_map_eq_zero_iff R _ K, hx, Algebra.smul_def]
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_3
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      I : FractionalIdeal (nonZeroDivisors R) K
      inst✝ : Nontrivial R
      hI : Ne I 0
      y : K
      y_mem : Membership.mem I y
      y_not_mem : Not (Membership.mem Bot.bot y)
      y_ne_zero : Ne y 0
      z : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      x : R
      hx : Eq ((algebraMap R K) x) (HSMul.hSMul (↑z) y)
      ⊢ Not (Eq (HMul.hMul ((algebraMap R K) ↑z) y) 0)
    -/
    exact mul_ne_zero (IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors z.2) y_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_3
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      I : FractionalIdeal (nonZeroDivisors R) K
      inst✝ : Nontrivial R
      hI : Ne I 0
      y : K
      y_mem : Membership.mem I y
      y_not_mem : Not (Membership.mem Bot.bot y)
      y_ne_zero : Ne y 0
      z : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      x : R
      hx : Eq ((algebraMap R K) x) (HSMul.hSMul (↑z) y)
      ⊢ Membership.mem I ((algebraMap R K) x)
    -/
  · rw [hx]
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_3
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      I : FractionalIdeal (nonZeroDivisors R) K
      inst✝ : Nontrivial R
      hI : Ne I 0
      y : K
      y_mem : Membership.mem I y
      y_not_mem : Not (Membership.mem Bot.bot y)
      y_ne_zero : Ne y 0
      z : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      x : R
      hx : Eq ((algebraMap R K) x) (HSMul.hSMul (↑z) y)
      ⊢ Membership.mem I (HSMul.hSMul (↑z) y)
    -/
    exact smul_mem _ _ y_mem
    /-
      🎉 no goals
    -/


theorem map_ne_zero [Nontrivial R] (hI : I ≠ 0) : I.map h ≠ 0 := by
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    K : Type u_3
    K' : Type u_4
    inst✝⁶ : Field K
    inst✝⁵ : Field K'
    inst✝⁴ : Algebra R K
    inst✝³ : IsFractionRing R K
    inst✝² : Algebra R K'
    inst✝¹ : IsFractionRing R K'
    I : FractionalIdeal (nonZeroDivisors R) K
    h : AlgHom R K K'
    inst✝ : Nontrivial R
    hI : Ne I 0
    ⊢ Ne (FractionalIdeal.map h I) 0
  -/
  obtain ⟨x, x_ne_zero, hx⟩ := exists_ne_zero_mem_isInteger hI
  /-
    case intro.intro
    R : Type u_1
    inst✝⁷ : CommRing R
    K : Type u_3
    K' : Type u_4
    inst✝⁶ : Field K
    inst✝⁵ : Field K'
    inst✝⁴ : Algebra R K
    inst✝³ : IsFractionRing R K
    inst✝² : Algebra R K'
    inst✝¹ : IsFractionRing R K'
    I : FractionalIdeal (nonZeroDivisors R) K
    h : AlgHom R K K'
    inst✝ : Nontrivial R
    hI : Ne I 0
    x : R
    x_ne_zero : Ne x 0
    hx : Membership.mem I ((algebraMap R K) x)
    ⊢ Ne (FractionalIdeal.map h I) 0
  -/
  contrapose! x_ne_zero with map_eq_zero
  /-
    case intro.intro
    R : Type u_1
    inst✝⁷ : CommRing R
    K : Type u_3
    K' : Type u_4
    inst✝⁶ : Field K
    inst✝⁵ : Field K'
    inst✝⁴ : Algebra R K
    inst✝³ : IsFractionRing R K
    inst✝² : Algebra R K'
    inst✝¹ : IsFractionRing R K'
    I : FractionalIdeal (nonZeroDivisors R) K
    h : AlgHom R K K'
    inst✝ : Nontrivial R
    hI : Ne I 0
    x : R
    hx : Membership.mem I ((algebraMap R K) x)
    map_eq_zero : Eq (FractionalIdeal.map h I) 0
    ⊢ Eq x 0
  -/
  refine IsFractionRing.to_map_eq_zero_iff.mp (eq_zero_iff.mp map_eq_zero _ (mem_map.mpr ?_))
  /-
    case intro.intro
    R : Type u_1
    inst✝⁷ : CommRing R
    K : Type u_3
    K' : Type u_4
    inst✝⁶ : Field K
    inst✝⁵ : Field K'
    inst✝⁴ : Algebra R K
    inst✝³ : IsFractionRing R K
    inst✝² : Algebra R K'
    inst✝¹ : IsFractionRing R K'
    I : FractionalIdeal (nonZeroDivisors R) K
    h : AlgHom R K K'
    inst✝ : Nontrivial R
    hI : Ne I 0
    x : R
    hx : Membership.mem I ((algebraMap R K) x)
    map_eq_zero : Eq (FractionalIdeal.map h I) 0
    ⊢ Exists fun x_1 => And (Membership.mem I x_1) (Eq (h x_1) ((algebraMap R K')  …
  -/
  exact ⟨algebraMap R K x, hx, h.commutes x⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem map_eq_zero_iff [Nontrivial R] : I.map h = 0 ↔ I = 0 :=
  ⟨not_imp_not.mp (map_ne_zero _), fun hI => hI.symm ▸ map_zero h⟩


theorem coeIdeal_injective : Function.Injective (fun (I : Ideal R) ↦ (I : FractionalIdeal R⁰ K)) :=
  coeIdeal_injective' le_rfl


theorem coeIdeal_inj {I J : Ideal R} :
    (I : FractionalIdeal R⁰ K) = (J : FractionalIdeal R⁰ K) ↔ I = J :=
  coeIdeal_inj' le_rfl


@[simp]
theorem coeIdeal_eq_zero {I : Ideal R} : (I : FractionalIdeal R⁰ K) = 0 ↔ I = ⊥ :=
  coeIdeal_eq_zero' le_rfl


theorem coeIdeal_ne_zero {I : Ideal R} : (I : FractionalIdeal R⁰ K) ≠ 0 ↔ I ≠ ⊥ :=
  coeIdeal_ne_zero' le_rfl


@[simp]
theorem coeIdeal_eq_one {I : Ideal R} : (I : FractionalIdeal R⁰ K) = 1 ↔ I = 1 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    K : Type u_3
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : Ideal R
    ⊢ Iff (Eq (↑I) 1) (Eq I 1)
  -/
  simpa only [Ideal.one_eq_top] using coeIdeal_inj
  /-
    🎉 no goals
  -/


theorem coeIdeal_ne_one {I : Ideal R} : (I : FractionalIdeal R⁰ K) ≠ 1 ↔ I ≠ 1 :=
  not_iff_not.mpr coeIdeal_eq_one


theorem num_eq_zero_iff [Nontrivial R] {I : FractionalIdeal R⁰ K} : I.num = 0 ↔ I = 0 :=
   ⟨fun h ↦ zero_of_num_eq_bot zero_not_mem_nonZeroDivisors h,
     fun h ↦ h ▸ num_zero_eq (IsFractionRing.injective R K)⟩


instance : Nontrivial (FractionalIdeal R₁⁰ K) :=
  ⟨⟨0, 1, fun h =>
      have this : (1 : K) ∈ (0 : FractionalIdeal R₁⁰ K) := by
        /-
          R : Type u_1
          inst✝⁵ : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝⁴ : CommRing P
          inst✝³ : Algebra R P
          R₁ : Type u_3
          inst✝² : CommRing R₁
          K : Type u_4
          inst✝¹ : Field K
          inst✝ : Algebra R₁ K
          h : Eq 0 1
          ⊢ Membership.mem 0 1
        -/
        rw [← (algebraMap R₁ K).map_one]
        /-
          R : Type u_1
          inst✝⁵ : CommRing R
          S : Submonoid R
          P : Type u_2
          inst✝⁴ : CommRing P
          inst✝³ : Algebra R P
          R₁ : Type u_3
          inst✝² : CommRing R₁
          K : Type u_4
          inst✝¹ : Field K
          inst✝ : Algebra R₁ K
          h : Eq 0 1
          ⊢ Membership.mem 0 ((algebraMap R₁ K) 1)
        -/
        simpa only [h] using coe_mem_one R₁⁰ 1
        /-
          🎉 no goals
        -/
      one_ne_zero ((mem_zero_iff _).mp this)⟩⟩


theorem ne_zero_of_mul_eq_one (I J : FractionalIdeal R₁⁰ K) (h : I * J = 1) : I ≠ 0 := fun hI =>
  zero_ne_one' (FractionalIdeal R₁⁰ K)
    (by
      /-
        R₁ : Type u_3
        inst✝² : CommRing R₁
        K : Type u_4
        inst✝¹ : Field K
        inst✝ : Algebra R₁ K
        I J : FractionalIdeal (nonZeroDivisors R₁) K
        h : Eq (HMul.hMul I J) 1
        hI : Eq I 0
        ⊢ Eq 0 1
      -/
      convert h
      /-
        case h.e'_2
        R₁ : Type u_3
        inst✝² : CommRing R₁
        K : Type u_4
        inst✝¹ : Field K
        inst✝ : Algebra R₁ K
        I J : FractionalIdeal (nonZeroDivisors R₁) K
        h : Eq (HMul.hMul I J) 1
        hI : Eq I 0
        ⊢ Eq 0 (HMul.hMul I J)
      -/
      simp [hI])
      /-
        🎉 no goals
      -/


theorem _root_.IsFractional.div_of_nonzero {I J : Submodule R₁ K} :
    IsFractional R₁⁰ I → IsFractional R₁⁰ J → J ≠ 0 → IsFractional R₁⁰ (I / J)
  | ⟨aI, haI, hI⟩, ⟨aJ, haJ, hJ⟩, h => by
    obtain ⟨y, mem_J, not_mem_zero⟩ :=
      SetLike.exists_of_lt (show 0 < J by simpa only using bot_lt_iff_ne_bot.mpr h)
    /-
      case intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : Submodule R₁ K
      aI : R₁
      haI : Membership.mem (nonZeroDivisors R₁) aI
      hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      aJ : R₁
      haJ : Membership.mem (nonZeroDivisors R₁) aJ
      hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      h : Ne J 0
      y : K
      mem_J : Membership.mem J y
      not_mem_zero : Not (Membership.mem 0 y)
      ⊢ IsFractional (nonZeroDivisors R₁) (HDiv.hDiv I J)
    -/
    obtain ⟨y', hy'⟩ := hJ y mem_J
    /-
      case intro.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : Submodule R₁ K
      aI : R₁
      haI : Membership.mem (nonZeroDivisors R₁) aI
      hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      aJ : R₁
      haJ : Membership.mem (nonZeroDivisors R₁) aJ
      hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      h : Ne J 0
      y : K
      mem_J : Membership.mem J y
      not_mem_zero : Not (Membership.mem 0 y)
      y' : R₁
      hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
      ⊢ IsFractional (nonZeroDivisors R₁) (HDiv.hDiv I J)
    -/
    use aI * y'
    /-
      case h
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : Submodule R₁ K
      aI : R₁
      haI : Membership.mem (nonZeroDivisors R₁) aI
      hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      aJ : R₁
      haJ : Membership.mem (nonZeroDivisors R₁) aJ
      hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      h : Ne J 0
      y : K
      mem_J : Membership.mem J y
      not_mem_zero : Not (Membership.mem 0 y)
      y' : R₁
      hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
      ⊢ And (Membership.mem (nonZeroDivisors R₁) (HMul.hMul aI y')) (∀ (b : K), Memb …
    -/
    constructor
      /-
        case h.left
        R₁ : Type u_3
        inst✝⁴ : CommRing R₁
        K : Type u_4
        inst✝³ : Field K
        inst✝² : Algebra R₁ K
        inst✝¹ : IsFractionRing R₁ K
        inst✝ : IsDomain R₁
        I J : Submodule R₁ K
        aI : R₁
        haI : Membership.mem (nonZeroDivisors R₁) aI
        hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        aJ : R₁
        haJ : Membership.mem (nonZeroDivisors R₁) aJ
        hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        h : Ne J 0
        y : K
        mem_J : Membership.mem J y
        not_mem_zero : Not (Membership.mem 0 y)
        y' : R₁
        hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
        ⊢ Membership.mem (nonZeroDivisors R₁) (HMul.hMul aI y')
      -/
    · apply (nonZeroDivisors R₁).mul_mem haI (mem_nonZeroDivisors_iff_ne_zero.mpr _)
      /-
        R₁ : Type u_3
        inst✝⁴ : CommRing R₁
        K : Type u_4
        inst✝³ : Field K
        inst✝² : Algebra R₁ K
        inst✝¹ : IsFractionRing R₁ K
        inst✝ : IsDomain R₁
        I J : Submodule R₁ K
        aI : R₁
        haI : Membership.mem (nonZeroDivisors R₁) aI
        hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        aJ : R₁
        haJ : Membership.mem (nonZeroDivisors R₁) aJ
        hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        h : Ne J 0
        y : K
        mem_J : Membership.mem J y
        not_mem_zero : Not (Membership.mem 0 y)
        y' : R₁
        hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
        ⊢ Ne y' 0
      -/
      intro y'_eq_zero
      have : algebraMap R₁ K aJ * y = 0 := by
        rw [← Algebra.smul_def, ← hy', y'_eq_zero, RingHom.map_zero]
      have y_zero :=
        (mul_eq_zero.mp this).resolve_left
          (mt ((injective_iff_map_eq_zero (algebraMap R₁ K)).1 (IsFractionRing.injective _ _) _)
            (mem_nonZeroDivisors_iff_ne_zero.mp haJ))
      /-
        R₁ : Type u_3
        inst✝⁴ : CommRing R₁
        K : Type u_4
        inst✝³ : Field K
        inst✝² : Algebra R₁ K
        inst✝¹ : IsFractionRing R₁ K
        inst✝ : IsDomain R₁
        I J : Submodule R₁ K
        aI : R₁
        haI : Membership.mem (nonZeroDivisors R₁) aI
        hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        aJ : R₁
        haJ : Membership.mem (nonZeroDivisors R₁) aJ
        hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        h : Ne J 0
        y : K
        mem_J : Membership.mem J y
        not_mem_zero : Not (Membership.mem 0 y)
        y' : R₁
        hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
        y'_eq_zero : Eq y' 0
        this : Eq (HMul.hMul ((algebraMap R₁ K) aJ) y) 0
        y_zero : Eq y 0
        ⊢ False
      -/
      apply not_mem_zero
      /-
        R₁ : Type u_3
        inst✝⁴ : CommRing R₁
        K : Type u_4
        inst✝³ : Field K
        inst✝² : Algebra R₁ K
        inst✝¹ : IsFractionRing R₁ K
        inst✝ : IsDomain R₁
        I J : Submodule R₁ K
        aI : R₁
        haI : Membership.mem (nonZeroDivisors R₁) aI
        hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        aJ : R₁
        haJ : Membership.mem (nonZeroDivisors R₁) aJ
        hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
        h : Ne J 0
        y : K
        mem_J : Membership.mem J y
        not_mem_zero : Not (Membership.mem 0 y)
        y' : R₁
        hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
        y'_eq_zero : Eq y' 0
        this : Eq (HMul.hMul ((algebraMap R₁ K) aJ) y) 0
        y_zero : Eq y 0
        ⊢ Membership.mem 0 y
      -/
      simpa
      /-
        🎉 no goals
      -/
    /-
      case h.right
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : Submodule R₁ K
      aI : R₁
      haI : Membership.mem (nonZeroDivisors R₁) aI
      hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      aJ : R₁
      haJ : Membership.mem (nonZeroDivisors R₁) aJ
      hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      h : Ne J 0
      y : K
      mem_J : Membership.mem J y
      not_mem_zero : Not (Membership.mem 0 y)
      y' : R₁
      hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
      ⊢ ∀ (b : K), Membership.mem (HDiv.hDiv I J) b → IsLocalization.IsInteger R₁ (H …
    -/
    intro b hb
    /-
      case h.right
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : Submodule R₁ K
      aI : R₁
      haI : Membership.mem (nonZeroDivisors R₁) aI
      hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      aJ : R₁
      haJ : Membership.mem (nonZeroDivisors R₁) aJ
      hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      h : Ne J 0
      y : K
      mem_J : Membership.mem J y
      not_mem_zero : Not (Membership.mem 0 y)
      y' : R₁
      hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
      b : K
      hb : Membership.mem (HDiv.hDiv I J) b
      ⊢ IsLocalization.IsInteger R₁ (HSMul.hSMul (HMul.hMul aI y') b)
    -/
    convert hI _ (hb _ (Submodule.smul_mem _ aJ mem_J)) using 1
    /-
      case h.e'_6
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : Submodule R₁ K
      aI : R₁
      haI : Membership.mem (nonZeroDivisors R₁) aI
      hI : ∀ (b : K), Membership.mem I b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      aJ : R₁
      haJ : Membership.mem (nonZeroDivisors R₁) aJ
      hJ : ∀ (b : K), Membership.mem J b → IsLocalization.IsInteger R₁ (HSMul.hSMul  …
      h : Ne J 0
      y : K
      mem_J : Membership.mem J y
      not_mem_zero : Not (Membership.mem 0 y)
      y' : R₁
      hy' : Eq ((algebraMap R₁ K) y') (HSMul.hSMul aJ y)
      b : K
      hb : Membership.mem (HDiv.hDiv I J) b
      ⊢ Eq (HSMul.hSMul (HMul.hMul aI y') b) (HSMul.hSMul aI (HMul.hMul b (HSMul.hSM …
    -/
    rw [← hy', mul_comm b, ← Algebra.smul_def, mul_smul]
    /-
      🎉 no goals
    -/


theorem fractional_div_of_nonzero {I J : FractionalIdeal R₁⁰ K} (h : J ≠ 0) :
    IsFractional R₁⁰ (I / J : Submodule R₁ K) :=
  I.isFractional.div_of_nonzero J.isFractional fun H =>
    h <| coeToSubmodule_injective <| H.trans coe_zero.symm


open Classical in
noncomputable instance : Div (FractionalIdeal R₁⁰ K) :=
  ⟨fun I J => if h : J = 0 then 0 else ⟨I / J, fractional_div_of_nonzero h⟩⟩


@[simp]
theorem div_zero {I : FractionalIdeal R₁⁰ K} : I / 0 = 0 :=
  dif_pos rfl


theorem div_nonzero {I J : FractionalIdeal R₁⁰ K} (h : J ≠ 0) :
    I / J = ⟨I / J, fractional_div_of_nonzero h⟩ :=
  dif_neg h


@[simp]
theorem coe_div {I J : FractionalIdeal R₁⁰ K} (hJ : J ≠ 0) :
    (↑(I / J) : Submodule R₁ K) = ↑I / (↑J : Submodule R₁ K) :=
  congr_arg _ (dif_neg hJ)


theorem mem_div_iff_of_nonzero {I J : FractionalIdeal R₁⁰ K} (h : J ≠ 0) {x} :
    x ∈ I / J ↔ ∀ y ∈ J, x * y ∈ I := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Ne J 0
    x : K
    ⊢ Iff (Membership.mem (HDiv.hDiv I J) x) (∀ (y : K), Membership.mem J y → Memb …
  -/
  rw [div_nonzero h]
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Ne J 0
    x : K
    ⊢ Iff (Membership.mem ⟨HDiv.hDiv ↑I ↑J, ⋯⟩ x) (∀ (y : K), Membership.mem J y → …
  -/
  exact Submodule.mem_div_iff_forall_mul_mem
  /-
    🎉 no goals
  -/


theorem mul_one_div_le_one {I : FractionalIdeal R₁⁰ K} : I * (1 / I) ≤ 1 := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    ⊢ LE.le (HMul.hMul I (HDiv.hDiv 1 I)) 1
  -/
  by_cases hI : I = 0
    /-
      case pos
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : Eq I 0
      ⊢ LE.le (HMul.hMul I (HDiv.hDiv 1 I)) 1
    -/
  · rw [hI, div_zero, mul_zero]
    /-
      case pos
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : Eq I 0
      ⊢ LE.le 0 1
    -/
    exact zero_le 1
    /-
      🎉 no goals
    -/
    /-
      case neg
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : Not (Eq I 0)
      ⊢ LE.le (HMul.hMul I (HDiv.hDiv 1 I)) 1
    -/
  · rw [← coe_le_coe, coe_mul, coe_div hI, coe_one]
    /-
      case neg
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : Not (Eq I 0)
      ⊢ LE.le (HMul.hMul (↑I) (HDiv.hDiv 1 ↑I)) 1
    -/
    apply Submodule.mul_one_div_le_one
    /-
      🎉 no goals
    -/


theorem le_self_mul_one_div {I : FractionalIdeal R₁⁰ K} (hI : I ≤ (1 : FractionalIdeal R₁⁰ K)) :
    I ≤ I * (1 / I) := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : LE.le I 1
    ⊢ LE.le I (HMul.hMul I (HDiv.hDiv 1 I))
  -/
  by_cases hI_nz : I = 0
    /-
      case pos
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : LE.le I 1
      hI_nz : Eq I 0
      ⊢ LE.le I (HMul.hMul I (HDiv.hDiv 1 I))
    -/
  · rw [hI_nz, div_zero, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : LE.le I 1
      hI_nz : Not (Eq I 0)
      ⊢ LE.le I (HMul.hMul I (HDiv.hDiv 1 I))
    -/
  · rw [← coe_le_coe, coe_mul, coe_div hI_nz, coe_one]
    /-
      case neg
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : LE.le I 1
      hI_nz : Not (Eq I 0)
      ⊢ LE.le (↑I) (HMul.hMul (↑I) (HDiv.hDiv 1 ↑I))
    -/
    rw [← coe_le_coe, coe_one] at hI
    /-
      case neg
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : LE.le (↑I) 1
      hI_nz : Not (Eq I 0)
      ⊢ LE.le (↑I) (HMul.hMul (↑I) (HDiv.hDiv 1 ↑I))
    -/
    exact Submodule.le_self_mul_one_div hI
    /-
      🎉 no goals
    -/


theorem le_div_iff_of_nonzero {I J J' : FractionalIdeal R₁⁰ K} (hJ' : J' ≠ 0) :
    I ≤ J / J' ↔ ∀ x ∈ I, ∀ y ∈ J', x * y ∈ J :=
  ⟨fun h _ hx => (mem_div_iff_of_nonzero hJ').mp (h hx), fun h x hx =>
    (mem_div_iff_of_nonzero hJ').mpr (h x hx)⟩


theorem le_div_iff_mul_le {I J J' : FractionalIdeal R₁⁰ K} (hJ' : J' ≠ 0) :
    I ≤ J / J' ↔ I * J' ≤ J := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J J' : FractionalIdeal (nonZeroDivisors R₁) K
    hJ' : Ne J' 0
    ⊢ Iff (LE.le I (HDiv.hDiv J J')) (LE.le (HMul.hMul I J') J)
  -/
  rw [div_nonzero hJ']
  -- Porting note: this used to be { convert; rw }, flipped the order.
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J J' : FractionalIdeal (nonZeroDivisors R₁) K
    hJ' : Ne J' 0
    ⊢ Iff (LE.le I ⟨HDiv.hDiv ↑J ↑J', ⋯⟩) (LE.le (HMul.hMul I J') J)
  -/
  rw [← coe_le_coe (I := I * J') (J := J), coe_mul]
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J J' : FractionalIdeal (nonZeroDivisors R₁) K
    hJ' : Ne J' 0
    ⊢ Iff (LE.le I ⟨HDiv.hDiv ↑J ↑J', ⋯⟩) (LE.le (HMul.hMul ↑I ↑J') ↑J)
  -/
  exact Submodule.le_div_iff_mul_le
  /-
    🎉 no goals
  -/


@[simp]
theorem div_one {I : FractionalIdeal R₁⁰ K} : I / 1 = I := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    ⊢ Eq (HDiv.hDiv I 1) I
  -/
  rw [div_nonzero (one_ne_zero' (FractionalIdeal R₁⁰ K))]
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    ⊢ Eq ⟨HDiv.hDiv ↑I ↑1, ⋯⟩ I
  -/
  ext
  /-
    case a
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    x✝ : K
    ⊢ Iff (Membership.mem ⟨HDiv.hDiv ↑I ↑1, ⋯⟩ x✝) (Membership.mem I x✝)
  -/
  constructor <;> intro h
    /-
      case a.mp
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      x✝ : K
      h : Membership.mem ⟨HDiv.hDiv ↑I ↑1, ⋯⟩ x✝
      ⊢ Membership.mem I x✝
    -/
  · simpa using mem_div_iff_forall_mul_mem.mp h 1 ((algebraMap R₁ K).map_one ▸ coe_mem_one R₁⁰ 1)
    /-
      🎉 no goals
    -/
    /-
      case a.mpr
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      x✝ : K
      h : Membership.mem I x✝
      ⊢ Membership.mem ⟨HDiv.hDiv ↑I ↑1, ⋯⟩ x✝
    -/
  · apply mem_div_iff_forall_mul_mem.mpr
    /-
      case a.mpr
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      x✝ : K
      h : Membership.mem I x✝
      ⊢ ∀ (y : K), Membership.mem (↑1) y → Membership.mem (↑I) (HMul.hMul x✝ y)
    -/
    rintro y ⟨y', _, rfl⟩
    -- Porting note: this used to be { convert; rw }, flipped the order.
    /-
      case a.mpr.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      x✝ : K
      h : Membership.mem I x✝
      y' : R₁
      left✝ : Membership.mem (↑Top.top) y'
      ⊢ Membership.mem (↑I) (HMul.hMul x✝ ((Algebra.linearMap R₁ K) y'))
    -/
    rw [mul_comm, Algebra.linearMap_apply, ← Algebra.smul_def]
    /-
      case a.mpr.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      x✝ : K
      h : Membership.mem I x✝
      y' : R₁
      left✝ : Membership.mem (↑Top.top) y'
      ⊢ Membership.mem (↑I) (HSMul.hSMul y' x✝)
    -/
    exact Submodule.smul_mem _ y' h
    /-
      🎉 no goals
    -/


theorem eq_one_div_of_mul_eq_one_right (I J : FractionalIdeal R₁⁰ K) (h : I * J = 1) :
    J = 1 / I := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    ⊢ Eq J (HDiv.hDiv 1 I)
  -/
  have hI : I ≠ 0 := ne_zero_of_mul_eq_one I J h
  suffices h' : I * (1 / I) = 1 from
    congr_arg Units.inv <| @Units.ext _ _ (Units.mkOfMulEqOne _ _ h) (Units.mkOfMulEqOne _ _ h') rfl
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ Eq (HMul.hMul I (HDiv.hDiv 1 I)) 1
  -/
  apply le_antisymm
    /-
      case a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      ⊢ LE.le (HMul.hMul I (HDiv.hDiv 1 I)) 1
    -/
  · apply mul_le.mpr _
    /-
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      ⊢ ∀ (i : K), Membership.mem I i → ∀ (j : K), Membership.mem (HDiv.hDiv 1 I) j  …
    -/
    intro x hx y hy
    /-
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      x : K
      hx : Membership.mem I x
      y : K
      hy : Membership.mem (HDiv.hDiv 1 I) y
      ⊢ Membership.mem 1 (HMul.hMul x y)
    -/
    rw [mul_comm]
    /-
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      x : K
      hx : Membership.mem I x
      y : K
      hy : Membership.mem (HDiv.hDiv 1 I) y
      ⊢ Membership.mem 1 (HMul.hMul y x)
    -/
    exact (mem_div_iff_of_nonzero hI).mp hy x hx
    /-
      🎉 no goals
    -/
  /-
    case a
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ LE.le 1 (HMul.hMul I (HDiv.hDiv 1 I))
  -/
  rw [← h]
  /-
    case a
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ LE.le (HMul.hMul I J) (HMul.hMul I (HDiv.hDiv (HMul.hMul I J) I))
  -/
  apply mul_left_mono I
  /-
    case a.a
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ LE.le J (HDiv.hDiv (HMul.hMul I J) I)
  -/
  apply (le_div_iff_of_nonzero hI).mpr _
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ ∀ (x : K), Membership.mem J x → ∀ (y : K), Membership.mem I y → Membership.m …
  -/
  intro y hy x hx
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    y : K
    hy : Membership.mem J y
    x : K
    hx : Membership.mem I x
    ⊢ Membership.mem (HMul.hMul I J) (HMul.hMul y x)
  -/
  rw [mul_comm]
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    y : K
    hy : Membership.mem J y
    x : K
    hx : Membership.mem I x
    ⊢ Membership.mem (HMul.hMul J I) (HMul.hMul y x)
  -/
  exact mul_mem_mul hy hx
  /-
    🎉 no goals
  -/


theorem mul_div_self_cancel_iff {I : FractionalIdeal R₁⁰ K} : I * (1 / I) = 1 ↔ ∃ J, I * J = 1 :=
                                          /-
                                            R₁ : Type u_3
                                            inst✝⁴ : CommRing R₁
                                            K : Type u_4
                                            inst✝³ : Field K
                                            inst✝² : Algebra R₁ K
                                            inst✝¹ : IsFractionRing R₁ K
                                            inst✝ : IsDomain R₁
                                            I : FractionalIdeal (nonZeroDivisors R₁) K
                                            x✝ : Exists fun J => Eq (HMul.hMul I J) 1
                                            J : FractionalIdeal (nonZeroDivisors R₁) K
                                            hJ : Eq (HMul.hMul I J) 1
                                            ⊢ Eq (HMul.hMul I (HDiv.hDiv 1 I)) 1
                                          -/
  ⟨fun h => ⟨1 / I, h⟩, fun ⟨J, hJ⟩ => by rwa [← eq_one_div_of_mul_eq_one_right I J hJ]⟩
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem map_div (I J : FractionalIdeal R₁⁰ K) (h : K ≃ₐ[R₁] K') :
    (I / J).map (h : K →ₐ[R₁] K') = I.map h / J.map h := by
  /-
    R₁ : Type u_3
    inst✝⁷ : CommRing R₁
    K : Type u_4
    inst✝⁶ : Field K
    inst✝⁵ : Algebra R₁ K
    inst✝⁴ : IsFractionRing R₁ K
    inst✝³ : IsDomain R₁
    K' : Type u_5
    inst✝² : Field K'
    inst✝¹ : Algebra R₁ K'
    inst✝ : IsFractionRing R₁ K'
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : AlgEquiv R₁ K K'
    ⊢ Eq (FractionalIdeal.map (↑h) (HDiv.hDiv I J)) (HDiv.hDiv (FractionalIdeal.ma …
  -/
  by_cases H : J = 0
    /-
      case pos
      R₁ : Type u_3
      inst✝⁷ : CommRing R₁
      K : Type u_4
      inst✝⁶ : Field K
      inst✝⁵ : Algebra R₁ K
      inst✝⁴ : IsFractionRing R₁ K
      inst✝³ : IsDomain R₁
      K' : Type u_5
      inst✝² : Field K'
      inst✝¹ : Algebra R₁ K'
      inst✝ : IsFractionRing R₁ K'
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : AlgEquiv R₁ K K'
      H : Eq J 0
      ⊢ Eq (FractionalIdeal.map (↑h) (HDiv.hDiv I J)) (HDiv.hDiv (FractionalIdeal.ma …
    -/
  · rw [H, div_zero, map_zero, div_zero]
    /-
      🎉 no goals
    -/
  · -- Porting note: `simp` wouldn't apply these lemmas so do them manually using `rw`
    /-
      case neg
      R₁ : Type u_3
      inst✝⁷ : CommRing R₁
      K : Type u_4
      inst✝⁶ : Field K
      inst✝⁵ : Algebra R₁ K
      inst✝⁴ : IsFractionRing R₁ K
      inst✝³ : IsDomain R₁
      K' : Type u_5
      inst✝² : Field K'
      inst✝¹ : Algebra R₁ K'
      inst✝ : IsFractionRing R₁ K'
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : AlgEquiv R₁ K K'
      H : Not (Eq J 0)
      ⊢ Eq (FractionalIdeal.map (↑h) (HDiv.hDiv I J)) (HDiv.hDiv (FractionalIdeal.ma …
    -/
    rw [← coeToSubmodule_inj, div_nonzero H, div_nonzero (map_ne_zero _ H)]
    /-
      case neg
      R₁ : Type u_3
      inst✝⁷ : CommRing R₁
      K : Type u_4
      inst✝⁶ : Field K
      inst✝⁵ : Algebra R₁ K
      inst✝⁴ : IsFractionRing R₁ K
      inst✝³ : IsDomain R₁
      K' : Type u_5
      inst✝² : Field K'
      inst✝¹ : Algebra R₁ K'
      inst✝ : IsFractionRing R₁ K'
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : AlgEquiv R₁ K K'
      H : Not (Eq J 0)
      ⊢ Eq ↑(FractionalIdeal.map ↑h ⟨HDiv.hDiv ↑I ↑J, ⋯⟩) ↑⟨HDiv.hDiv ↑(FractionalId …
    -/
    simp [Submodule.map_div]
    /-
      🎉 no goals
    -/

-- Porting note: doesn't need to be @[simp] because this follows from `map_one` and `map_div`

theorem map_one_div (I : FractionalIdeal R₁⁰ K) (h : K ≃ₐ[R₁] K') :
                                                      /-
                                                        R₁ : Type u_3
                                                        inst✝⁷ : CommRing R₁
                                                        K : Type u_4
                                                        inst✝⁶ : Field K
                                                        inst✝⁵ : Algebra R₁ K
                                                        inst✝⁴ : IsFractionRing R₁ K
                                                        inst✝³ : IsDomain R₁
                                                        K' : Type u_5
                                                        inst✝² : Field K'
                                                        inst✝¹ : Algebra R₁ K'
                                                        inst✝ : IsFractionRing R₁ K'
                                                        I : FractionalIdeal (nonZeroDivisors R₁) K
                                                        h : AlgEquiv R₁ K K'
                                                        ⊢ Eq (FractionalIdeal.map (↑h) (HDiv.hDiv 1 I)) (HDiv.hDiv 1 (FractionalIdeal. …
                                                      -/
    (1 / I).map (h : K →ₐ[R₁] K') = 1 / I.map h := by rw [map_div, map_one]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem eq_zero_or_one (I : FractionalIdeal K⁰ L) : I = 0 ∨ I = 1 := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsFractionRing K L
    I : FractionalIdeal (nonZeroDivisors K) L
    ⊢ Or (Eq I 0) (Eq I 1)
  -/
  rw [or_iff_not_imp_left]
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsFractionRing K L
    I : FractionalIdeal (nonZeroDivisors K) L
    ⊢ Not (Eq I 0) → Eq I 1
  -/
  intro hI
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsFractionRing K L
    I : FractionalIdeal (nonZeroDivisors K) L
    hI : Not (Eq I 0)
    ⊢ Eq I 1
  -/
  simp_rw [@SetLike.ext_iff _ _ _ I 1, mem_one_iff]
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsFractionRing K L
    I : FractionalIdeal (nonZeroDivisors K) L
    hI : Not (Eq I 0)
    ⊢ ∀ (x : L), Iff (Membership.mem I x) (Exists fun x' => Eq ((algebraMap K L) x …
  -/
  intro x
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsFractionRing K L
    I : FractionalIdeal (nonZeroDivisors K) L
    hI : Not (Eq I 0)
    x : L
    ⊢ Iff (Membership.mem I x) (Exists fun x' => Eq ((algebraMap K L) x') x)
  -/
  constructor
    /-
      case mp
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      x : L
      ⊢ Membership.mem I x → Exists fun x' => Eq ((algebraMap K L) x') x
    -/
  · intro x_mem
    /-
      case mp
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      x : L
      x_mem : Membership.mem I x
      ⊢ Exists fun x' => Eq ((algebraMap K L) x') x
    -/
    obtain ⟨n, d, rfl⟩ := IsLocalization.mk'_surjective K⁰ x
    /-
      case mp.intro.intro
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      n : K
      d : Subtype fun x => Membership.mem (nonZeroDivisors K) x
      x_mem : Membership.mem I (IsLocalization.mk' L n d)
      ⊢ Exists fun x' => Eq ((algebraMap K L) x') (IsLocalization.mk' L n d)
    -/
    refine ⟨n / d, ?_⟩
    /-
      case mp.intro.intro
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      n : K
      d : Subtype fun x => Membership.mem (nonZeroDivisors K) x
      x_mem : Membership.mem I (IsLocalization.mk' L n d)
      ⊢ Eq ((algebraMap K L) (HDiv.hDiv n ↑d)) (IsLocalization.mk' L n d)
    -/
    rw [map_div₀, IsFractionRing.mk'_eq_div]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      x : L
      ⊢ (Exists fun x' => Eq ((algebraMap K L) x') x) → Membership.mem I x
    -/
  · rintro ⟨x, rfl⟩
    /-
      case mpr.intro
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      x : K
      ⊢ Membership.mem I ((algebraMap K L) x)
    -/
    obtain ⟨y, y_ne, y_mem⟩ := exists_ne_zero_mem_isInteger hI
    /-
      case mpr.intro.intro.intro
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      x y : K
      y_ne : Ne y 0
      y_mem : Membership.mem I ((algebraMap K L) y)
      ⊢ Membership.mem I ((algebraMap K L) x)
    -/
    rw [← div_mul_cancel₀ x y_ne, RingHom.map_mul, ← Algebra.smul_def]
    /-
      case mpr.intro.intro.intro
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsFractionRing K L
      I : FractionalIdeal (nonZeroDivisors K) L
      hI : Not (Eq I 0)
      x y : K
      y_ne : Ne y 0
      y_mem : Membership.mem I ((algebraMap K L) y)
      ⊢ Membership.mem I (HSMul.hSMul (HDiv.hDiv x y) ((algebraMap K L) y))
    -/
    exact smul_mem (M := L) I (x / y) y_mem
    /-
      🎉 no goals
    -/


theorem eq_zero_or_one_of_isField (hF : IsField R₁) (I : FractionalIdeal R₁⁰ K) : I = 0 ∨ I = 1 :=
  letI : Field R₁ := hF.toField
  eq_zero_or_one I


/-- `FractionalIdeal.span_finset R₁ s f` is the fractional ideal of `R₁` generated by `f '' s`. -/
-- Porting note: `@[simps]` generated a `Subtype.val` coercion instead of a
-- `FractionalIdeal.coeToSubmodule` coercion
def spanFinset {ι : Type*} (s : Finset ι) (f : ι → K) : FractionalIdeal R₁⁰ K :=
  ⟨Submodule.span R₁ (f '' s), by
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      R₁ : Type u_3
      inst✝³ : CommRing R₁
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      ι : Type u_5
      s : Finset ι
      f : ι → K
      ⊢ IsFractional (nonZeroDivisors R₁) (Submodule.span R₁ (Set.image f ↑s))
    -/
    obtain ⟨a', ha'⟩ := IsLocalization.exist_integer_multiples R₁⁰ s f
    /-
      case intro
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      R₁ : Type u_3
      inst✝³ : CommRing R₁
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      ι : Type u_5
      s : Finset ι
      f : ι → K
      a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
      ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
      ⊢ IsFractional (nonZeroDivisors R₁) (Submodule.span R₁ (Set.image f ↑s))
    -/
    refine ⟨a', a'.2, fun x hx => Submodule.span_induction ?_ ?_ ?_ ?_ hx⟩
      /-
        case intro.refine_1
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x : K
        hx : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        ⊢ ∀ (x : K), Membership.mem (Set.image f ↑s) x → IsLocalization.IsInteger R₁ ( …
      -/
    · rintro _ ⟨i, hi, rfl⟩
      /-
        case intro.refine_1.intro.intro
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x : K
        hx : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        i : ι
        hi : Membership.mem (↑s) i
        ⊢ IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') (f i))
      -/
      exact ha' i hi
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x : K
        hx : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        ⊢ IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') 0)
      -/
    · rw [smul_zero]
      /-
        case intro.refine_2
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x : K
        hx : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        ⊢ IsLocalization.IsInteger R₁ 0
      -/
      exact IsLocalization.isInteger_zero
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_3
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x : K
        hx : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        ⊢ ∀ (x y : K), Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x → Members …
      -/
    · intro x y _ _ hx hy
      /-
        case intro.refine_3
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x✝ : K
        hx✝¹ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x✝
        x y : K
        hx✝ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        hy✝ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) y
        hx : IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') x)
        hy : IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') y)
        ⊢ IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') (HAdd.hAdd x y))
      -/
      rw [smul_add]
      /-
        case intro.refine_3
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x✝ : K
        hx✝¹ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x✝
        x y : K
        hx✝ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        hy✝ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) y
        hx : IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') x)
        hy : IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') y)
        ⊢ IsLocalization.IsInteger R₁ (HAdd.hAdd (HSMul.hSMul (↑a') x) (HSMul.hSMul (↑ …
      -/
      exact IsLocalization.isInteger_add hx hy
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_4
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x : K
        hx : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        ⊢ ∀ (a : R₁) (x : K), Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x →  …
      -/
    · intro c x _ hx
      /-
        case intro.refine_4
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x✝ : K
        hx✝¹ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x✝
        c : R₁
        x : K
        hx✝ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        hx : IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') x)
        ⊢ IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') (HSMul.hSMul c x))
      -/
      rw [smul_comm]
      /-
        case intro.refine_4
        R : Type u_1
        inst✝⁶ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝⁵ : CommRing P
        inst✝⁴ : Algebra R P
        R₁ : Type u_3
        inst✝³ : CommRing R₁
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Algebra R₁ K
        inst✝ : IsFractionRing R₁ K
        ι : Type u_5
        s : Finset ι
        f : ι → K
        a' : Subtype fun x => Membership.mem (nonZeroDivisors R₁) x
        ha' : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R₁ (HSMul.hSMul …
        x✝ : K
        hx✝¹ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x✝
        c : R₁
        x : K
        hx✝ : Membership.mem (Submodule.span R₁ (Set.image f ↑s)) x
        hx : IsLocalization.IsInteger R₁ (HSMul.hSMul (↑a') x)
        ⊢ IsLocalization.IsInteger R₁ (HSMul.hSMul c (HSMul.hSMul (↑a') x))
      -/
      exact IsLocalization.isInteger_smul hx⟩
      /-
        🎉 no goals
      -/


@[simp] lemma spanFinset_coe {ι : Type*} (s : Finset ι) (f : ι → K) :
    (spanFinset R₁ s f : Submodule R₁ K) = Submodule.span R₁ (f '' s) :=
  rfl


@[simp]
theorem spanFinset_eq_zero {ι : Type*} {s : Finset ι} {f : ι → K} :
    spanFinset R₁ s f = 0 ↔ ∀ j ∈ s, f j = 0 := by
  simp only [← coeToSubmodule_inj, spanFinset_coe, coe_zero, Submodule.span_eq_bot,
    Set.mem_image, Finset.mem_coe, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]


theorem spanFinset_ne_zero {ι : Type*} {s : Finset ι} {f : ι → K} :
                                                   /-
                                                     R₁ : Type u_3
                                                     inst✝³ : CommRing R₁
                                                     K : Type u_4
                                                     inst✝² : Field K
                                                     inst✝¹ : Algebra R₁ K
                                                     inst✝ : IsFractionRing R₁ K
                                                     ι : Type u_5
                                                     s : Finset ι
                                                     f : ι → K
                                                     ⊢ Iff (Ne (FractionalIdeal.spanFinset R₁ s f) 0) (Exists fun j => And (Members …
                                                   -/
    spanFinset R₁ s f ≠ 0 ↔ ∃ j ∈ s, f j ≠ 0 := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem isFractional_span_singleton (x : P) : IsFractional S (span R {x} : Submodule R P) :=
  let ⟨a, ha⟩ := exists_integer_multiple S x
  isFractional_span_iff.mpr ⟨a, a.2, fun _ hx' => (Set.mem_singleton_iff.mp hx').symm ▸ ha⟩


/-- `spanSingleton x` is the fractional ideal generated by `x` if `0 ∉ S` -/
irreducible_def spanSingleton (x : P) : FractionalIdeal S P :=
  ⟨span R {x}, isFractional_span_singleton x⟩

-- local attribute [semireducible] span_singleton

@[simp]
theorem coe_spanSingleton (x : P) : (spanSingleton S x : Submodule R P) = span R {x} := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    ⊢ Eq (↑(FractionalIdeal.spanSingleton S x)) (Submodule.span R (Singleton.singl …
  -/
  rw [spanSingleton]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    ⊢ Eq (↑⟨Submodule.span R (Singleton.singleton x), ⋯⟩) (Submodule.span R (Singl …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_spanSingleton {x y : P} : x ∈ spanSingleton S y ↔ ∃ z : R, z • y = x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x y : P
    ⊢ Iff (Membership.mem (FractionalIdeal.spanSingleton S y) x) (Exists fun z =>  …
  -/
  rw [spanSingleton]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x y : P
    ⊢ Iff (Membership.mem ⟨Submodule.span R (Singleton.singleton y), ⋯⟩ x) (Exists …
  -/
  exact Submodule.mem_span_singleton
  /-
    🎉 no goals
  -/


theorem mem_spanSingleton_self (x : P) : x ∈ spanSingleton S x :=
  (mem_spanSingleton S).mpr ⟨1, one_smul _ _⟩


variable (P) in
/-- A version of `FractionalIdeal.den_mul_self_eq_num` in terms of fractional ideals. -/
theorem den_mul_self_eq_num' (I : FractionalIdeal S P) :
    spanSingleton S (algebraMap R P I.den) * I = I.num := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ Eq (HMul.hMul (FractionalIdeal.spanSingleton S ((algebraMap R P) ↑I.den)) I) …
  -/
  apply coeToSubmodule_injective
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ Eq ((fun I => ↑I) (HMul.hMul (FractionalIdeal.spanSingleton S ((algebraMap R …
  -/
  dsimp only
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ Eq ↑(HMul.hMul (FractionalIdeal.spanSingleton S ((algebraMap R P) ↑I.den)) I …
  -/
  rw [coe_mul, ← smul_eq_mul, coe_spanSingleton, smul_eq_mul, Submodule.span_singleton_mul]
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ Eq (HSMul.hSMul ((algebraMap R P) ↑I.den) ↑I) ↑↑I.num
  -/
  convert I.den_mul_self_eq_num using 1
  /-
    case h.e'_2
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ Eq (HSMul.hSMul ((algebraMap R P) ↑I.den) ↑I) (HSMul.hSMul I.den ↑I)
  -/
  ext
  /-
    case h.e'_2.h
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    x✝ : P
    ⊢ Iff (Membership.mem (HSMul.hSMul ((algebraMap R P) ↑I.den) ↑I) x✝) (Membersh …
  -/
  erw [Set.mem_smul_set, Set.mem_smul_set]
  /-
    case h.e'_2.h
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    x✝ : P
    ⊢ Iff (Exists fun y => And (Membership.mem (↑↑I) y) (Eq (HSMul.hSMul ((algebra …
  -/
  simp [Algebra.smul_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanSingleton_le_iff_mem {x : P} {I : FractionalIdeal S P} :
    spanSingleton S x ≤ I ↔ x ∈ I := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    I : FractionalIdeal S P
    ⊢ Iff (LE.le (FractionalIdeal.spanSingleton S x) I) (Membership.mem I x)
  -/
  rw [← coe_le_coe, coe_spanSingleton, Submodule.span_singleton_le_iff_mem, mem_coe]
  /-
    🎉 no goals
  -/


theorem spanSingleton_eq_spanSingleton [NoZeroSMulDivisors R P] {x y : P} :
    spanSingleton S x = spanSingleton S y ↔ ∃ z : Rˣ, z • x = y := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    inst✝¹ : IsLocalization S P
    inst✝ : NoZeroSMulDivisors R P
    x y : P
    ⊢ Iff (Eq (FractionalIdeal.spanSingleton S x) (FractionalIdeal.spanSingleton S …
  -/
  rw [← Submodule.span_singleton_eq_span_singleton, spanSingleton, spanSingleton]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    inst✝¹ : IsLocalization S P
    inst✝ : NoZeroSMulDivisors R P
    x y : P
    ⊢ Iff (Eq ⟨Submodule.span R (Singleton.singleton x), ⋯⟩ ⟨Submodule.span R (Sin …
  -/
  exact Subtype.mk_eq_mk
  /-
    🎉 no goals
  -/


theorem eq_spanSingleton_of_principal (I : FractionalIdeal S P) [IsPrincipal (I : Submodule R P)] :
    I = spanSingleton S (generator (I : Submodule R P)) := by
  -- Porting note: this used to be `coeToSubmodule_injective (span_singleton_generator ↑I).symm`
  -- but Lean 4 struggled to unify everything. Turned it into an explicit `rw`.
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝³ : CommRing P
    inst✝² : Algebra R P
    inst✝¹ : IsLocalization S P
    I : FractionalIdeal S P
    inst✝ : (↑I).IsPrincipal
    ⊢ Eq I (FractionalIdeal.spanSingleton S (Submodule.IsPrincipal.generator ↑I))
  -/
  rw [spanSingleton, ← coeToSubmodule_inj, coe_mk, span_singleton_generator]
  /-
    🎉 no goals
  -/


theorem isPrincipal_iff (I : FractionalIdeal S P) :
    IsPrincipal (I : Submodule R P) ↔ ∃ x, I = spanSingleton S x :=
  ⟨fun _ => ⟨generator (I : Submodule R P), eq_spanSingleton_of_principal I⟩,
    fun ⟨x, hx⟩ => { principal' := ⟨x, Eq.trans (congr_arg _ hx) (coe_spanSingleton _ x)⟩ }⟩


@[simp]
theorem spanSingleton_zero : spanSingleton S (0 : P) = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    ⊢ Eq (FractionalIdeal.spanSingleton S 0) 0
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x✝ : P
    ⊢ Iff (Membership.mem (FractionalIdeal.spanSingleton S 0) x✝) (Membership.mem  …
  -/
  simp [Submodule.mem_span_singleton, eq_comm]
  /-
    🎉 no goals
  -/


theorem spanSingleton_eq_zero_iff {y : P} : spanSingleton S y = 0 ↔ y = 0 :=
  ⟨fun h =>
                       /-
                         R : Type u_1
                         inst✝³ : CommRing R
                         S : Submonoid R
                         P : Type u_2
                         inst✝² : CommRing P
                         inst✝¹ : Algebra R P
                         inst✝ : IsLocalization S P
                         y : P
                         h : Eq (FractionalIdeal.spanSingleton S y) 0
                         ⊢ Eq (Submodule.span R (Singleton.singleton y)) Bot.bot
                       -/
    span_eq_bot.mp (by simpa using congr_arg Subtype.val h : span R {y} = ⊥) y (mem_singleton y),
                       /-
                         🎉 no goals
                       -/
                /-
                  R : Type u_1
                  inst✝³ : CommRing R
                  S : Submonoid R
                  P : Type u_2
                  inst✝² : CommRing P
                  inst✝¹ : Algebra R P
                  inst✝ : IsLocalization S P
                  y : P
                  h : Eq y 0
                  ⊢ Eq (FractionalIdeal.spanSingleton S y) 0
                -/
    fun h => by simp [h]⟩
                /-
                  🎉 no goals
                -/


theorem spanSingleton_ne_zero_iff {y : P} : spanSingleton S y ≠ 0 ↔ y ≠ 0 :=
  not_congr spanSingleton_eq_zero_iff


@[simp]
theorem spanSingleton_one : spanSingleton S (1 : P) = 1 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    ⊢ Eq (FractionalIdeal.spanSingleton S 1) 1
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x✝ : P
    ⊢ Iff (Membership.mem (FractionalIdeal.spanSingleton S 1) x✝) (Membership.mem  …
  -/
  refine (mem_spanSingleton S).trans ((exists_congr ?_).trans (mem_one_iff S).symm)
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x✝ : P
    ⊢ ∀ (a : R), Iff (Eq (HSMul.hSMul a 1) x✝) (Eq ((algebraMap R P) a) x✝)
  -/
  intro x'
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x✝ : P
    x' : R
    ⊢ Iff (Eq (HSMul.hSMul x' 1) x✝) (Eq ((algebraMap R P) x') x✝)
  -/
  rw [Algebra.smul_def, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanSingleton_mul_spanSingleton (x y : P) :
    spanSingleton S x * spanSingleton S y = spanSingleton S (x * y) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x y : P
    ⊢ Eq (HMul.hMul (FractionalIdeal.spanSingleton S x) (FractionalIdeal.spanSingl …
  -/
  apply coeToSubmodule_injective
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x y : P
    ⊢ Eq ((fun I => ↑I) (HMul.hMul (FractionalIdeal.spanSingleton S x) (Fractional …
  -/
  simp only [coe_mul, coe_spanSingleton, span_mul_span, singleton_mul_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanSingleton_pow (x : P) (n : ℕ) : spanSingleton S x ^ n = spanSingleton S (x ^ n) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    n : Nat
    ⊢ Eq (HPow.hPow (FractionalIdeal.spanSingleton S x) n) (FractionalIdeal.spanSi …
  -/
  induction' n with n hn
    /-
      case zero
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      ⊢ Eq (HPow.hPow (FractionalIdeal.spanSingleton S x) 0) (FractionalIdeal.spanSi …
    -/
  · rw [pow_zero, pow_zero, spanSingleton_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      n : Nat
      hn : Eq (HPow.hPow (FractionalIdeal.spanSingleton S x) n) (FractionalIdeal.spa …
      ⊢ Eq (HPow.hPow (FractionalIdeal.spanSingleton S x) (HAdd.hAdd n 1)) (Fraction …
    -/
  · rw [pow_succ, hn, spanSingleton_mul_spanSingleton, pow_succ]
    /-
      🎉 no goals
    -/


@[simp]
theorem coeIdeal_span_singleton (x : R) :
    (↑(Ideal.span {x} : Ideal R) : FractionalIdeal S P) = spanSingleton S (algebraMap R P x) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : R
    ⊢ Eq (↑(Ideal.span (Singleton.singleton x))) (FractionalIdeal.spanSingleton S  …
  -/
  ext y
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : R
    y : P
    ⊢ Iff (Membership.mem (↑(Ideal.span (Singleton.singleton x))) y) (Membership.m …
  -/
  refine (mem_coeIdeal S).trans (Iff.trans ?_ (mem_spanSingleton S).symm)
  /-
    case a
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : R
    y : P
    ⊢ Iff (Exists fun x' => And (Membership.mem (Ideal.span (Singleton.singleton x …
  -/
  constructor
    /-
      case a.mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : R
      y : P
      ⊢ (Exists fun x' => And (Membership.mem (Ideal.span (Singleton.singleton x)) x …
    -/
  · rintro ⟨y', hy', rfl⟩
    /-
      case a.mp.intro.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x y' : R
      hy' : Membership.mem (Ideal.span (Singleton.singleton x)) y'
      ⊢ Exists fun z => Eq (HSMul.hSMul z ((algebraMap R P) x)) ((algebraMap R P) y')
    -/
    obtain ⟨x', rfl⟩ := Submodule.mem_span_singleton.mp hy'
    /-
      case a.mp.intro.intro.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x x' : R
      hy' : Membership.mem (Ideal.span (Singleton.singleton x)) (HSMul.hSMul x' x)
      ⊢ Exists fun z => Eq (HSMul.hSMul z ((algebraMap R P) x)) ((algebraMap R P) (H …
    -/
    use x'
    /-
      case h
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x x' : R
      hy' : Membership.mem (Ideal.span (Singleton.singleton x)) (HSMul.hSMul x' x)
      ⊢ Eq (HSMul.hSMul x' ((algebraMap R P) x)) ((algebraMap R P) (HSMul.hSMul x' x))
    -/
    rw [smul_eq_mul, RingHom.map_mul, Algebra.smul_def]
    /-
      🎉 no goals
    -/
    /-
      case a.mpr
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : R
      y : P
      ⊢ (Exists fun z => Eq (HSMul.hSMul z ((algebraMap R P) x)) y) → Exists fun x'  …
    -/
  · rintro ⟨y', rfl⟩
    /-
      case a.mpr.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x y' : R
      ⊢ Exists fun x' => And (Membership.mem (Ideal.span (Singleton.singleton x)) x' …
    -/
    refine ⟨y' * x, Submodule.mem_span_singleton.mpr ⟨y', rfl⟩, ?_⟩
    /-
      case a.mpr.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x y' : R
      ⊢ Eq ((algebraMap R P) (HMul.hMul y' x)) (HSMul.hSMul y' ((algebraMap R P) x))
    -/
    rw [RingHom.map_mul, Algebra.smul_def]
    /-
      🎉 no goals
    -/


@[simp]
theorem canonicalEquiv_spanSingleton {P'} [CommRing P'] [Algebra R P'] [IsLocalization S P']
    (x : P) :
    canonicalEquiv S P P' (spanSingleton S x) =
      spanSingleton S
        (IsLocalization.map P' (RingHom.id R)
          (fun y (hy : y ∈ S) => show RingHom.id R y ∈ S from hy) x) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    inst✝³ : IsLocalization S P
    P' : Type u_5
    inst✝² : CommRing P'
    inst✝¹ : Algebra R P'
    inst✝ : IsLocalization S P'
    x : P
    ⊢ Eq ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.spanSingleton S …
  -/
  apply SetLike.ext_iff.mpr
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    inst✝³ : IsLocalization S P
    P' : Type u_5
    inst✝² : CommRing P'
    inst✝¹ : Algebra R P'
    inst✝ : IsLocalization S P'
    x : P
    ⊢ ∀ (x_1 : P'), Iff (Membership.mem ((FractionalIdeal.canonicalEquiv S P P') ( …
  -/
  intro y
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝⁵ : CommRing P
    inst✝⁴ : Algebra R P
    inst✝³ : IsLocalization S P
    P' : Type u_5
    inst✝² : CommRing P'
    inst✝¹ : Algebra R P'
    inst✝ : IsLocalization S P'
    x : P
    y : P'
    ⊢ Iff (Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdea …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      y : P'
      h : Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.s …
      ⊢ Membership.mem (FractionalIdeal.spanSingleton S ((IsLocalization.map P' (Rin …
    -/
  · rw [mem_spanSingleton]
    /-
      case mp
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      y : P'
      h : Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.s …
      ⊢ Exists fun z => Eq (HSMul.hSMul z ((IsLocalization.map P' (RingHom.id R) ⋯)  …
    -/
    obtain ⟨x', hx', rfl⟩ := (mem_canonicalEquiv_apply _ _ _).mp h
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x x' : P
      hx' : Membership.mem (FractionalIdeal.spanSingleton S x) x'
      h : Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.s …
      ⊢ Exists fun z => Eq (HSMul.hSMul z ((IsLocalization.map P' (RingHom.id R) ⋯)  …
    -/
    obtain ⟨z, rfl⟩ := (mem_spanSingleton _).mp hx'
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      z : R
      hx' : Membership.mem (FractionalIdeal.spanSingleton S x) (HSMul.hSMul z x)
      h : Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.s …
      ⊢ Exists fun z_1 => Eq (HSMul.hSMul z_1 ((IsLocalization.map P' (RingHom.id R) …
    -/
    use z
    /-
      case h
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      z : R
      hx' : Membership.mem (FractionalIdeal.spanSingleton S x) (HSMul.hSMul z x)
      h : Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.s …
      ⊢ Eq (HSMul.hSMul z ((IsLocalization.map P' (RingHom.id R) ⋯) x)) ((IsLocaliza …
    -/
    rw [IsLocalization.map_smul, RingHom.id_apply]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      y : P'
      h : Membership.mem (FractionalIdeal.spanSingleton S ((IsLocalization.map P' (R …
      ⊢ Membership.mem ((FractionalIdeal.canonicalEquiv S P P') (FractionalIdeal.spa …
    -/
  · rw [mem_canonicalEquiv_apply]
    /-
      case mpr
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      y : P'
      h : Membership.mem (FractionalIdeal.spanSingleton S ((IsLocalization.map P' (R …
      ⊢ Exists fun y_1 => And (Membership.mem (FractionalIdeal.spanSingleton S x) y_ …
    -/
    obtain ⟨z, rfl⟩ := (mem_spanSingleton _).mp h
    /-
      case mpr.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      z : R
      h : Membership.mem (FractionalIdeal.spanSingleton S ((IsLocalization.map P' (R …
      ⊢ Exists fun y => And (Membership.mem (FractionalIdeal.spanSingleton S x) y) ( …
    -/
    use z • x
    /-
      case h
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      z : R
      h : Membership.mem (FractionalIdeal.spanSingleton S ((IsLocalization.map P' (R …
      ⊢ And (Membership.mem (FractionalIdeal.spanSingleton S x) (HSMul.hSMul z x)) ( …
    -/
    use (mem_spanSingleton _).mpr ⟨z, rfl⟩
    /-
      case right
      R : Type u_1
      inst✝⁶ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝⁵ : CommRing P
      inst✝⁴ : Algebra R P
      inst✝³ : IsLocalization S P
      P' : Type u_5
      inst✝² : CommRing P'
      inst✝¹ : Algebra R P'
      inst✝ : IsLocalization S P'
      x : P
      z : R
      h : Membership.mem (FractionalIdeal.spanSingleton S ((IsLocalization.map P' (R …
      ⊢ Eq ((IsLocalization.map P' (RingHom.id R) ⋯) (HSMul.hSMul z x)) (HSMul.hSMul …
    -/
    simp [IsLocalization.map_smul]
    /-
      🎉 no goals
    -/


theorem mem_singleton_mul {x y : P} {I : FractionalIdeal S P} :
    y ∈ spanSingleton S x * I ↔ ∃ y' ∈ I, y = x * y' := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x y : P
    I : FractionalIdeal S P
    ⊢ Iff (Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y) (Ex …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x y : P
      I : FractionalIdeal S P
      ⊢ Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y → Exists  …
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x y : P
      I : FractionalIdeal S P
      h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y
      ⊢ Exists fun y' => And (Membership.mem I y') (Eq y (HMul.hMul x y'))
    -/
    refine FractionalIdeal.mul_induction_on h ?_ ?_
      /-
        case mp.refine_1
        R : Type u_1
        inst✝³ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝² : CommRing P
        inst✝¹ : Algebra R P
        inst✝ : IsLocalization S P
        x y : P
        I : FractionalIdeal S P
        h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y
        ⊢ ∀ (i : P), Membership.mem (FractionalIdeal.spanSingleton S x) i → ∀ (j : P), …
      -/
    · intro x' hx' y' hy'
      /-
        case mp.refine_1
        R : Type u_1
        inst✝³ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝² : CommRing P
        inst✝¹ : Algebra R P
        inst✝ : IsLocalization S P
        x y : P
        I : FractionalIdeal S P
        h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y
        x' : P
        hx' : Membership.mem (FractionalIdeal.spanSingleton S x) x'
        y' : P
        hy' : Membership.mem I y'
        ⊢ Exists fun y'_1 => And (Membership.mem I y'_1) (Eq (HMul.hMul x' y') (HMul.h …
      -/
      obtain ⟨a, ha⟩ := (mem_spanSingleton S).mp hx'
      /-
        case mp.refine_1.intro
        R : Type u_1
        inst✝³ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝² : CommRing P
        inst✝¹ : Algebra R P
        inst✝ : IsLocalization S P
        x y : P
        I : FractionalIdeal S P
        h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y
        x' : P
        hx' : Membership.mem (FractionalIdeal.spanSingleton S x) x'
        y' : P
        hy' : Membership.mem I y'
        a : R
        ha : Eq (HSMul.hSMul a x) x'
        ⊢ Exists fun y'_1 => And (Membership.mem I y'_1) (Eq (HMul.hMul x' y') (HMul.h …
      -/
      use a • y', Submodule.smul_mem (I : Submodule R P) a hy'
      /-
        case right
        R : Type u_1
        inst✝³ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝² : CommRing P
        inst✝¹ : Algebra R P
        inst✝ : IsLocalization S P
        x y : P
        I : FractionalIdeal S P
        h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y
        x' : P
        hx' : Membership.mem (FractionalIdeal.spanSingleton S x) x'
        y' : P
        hy' : Membership.mem I y'
        a : R
        ha : Eq (HSMul.hSMul a x) x'
        ⊢ Eq (HMul.hMul x' y') (HMul.hMul x (HSMul.hSMul a y'))
      -/
      rw [← ha, Algebra.mul_smul_comm, Algebra.smul_mul_assoc]
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        R : Type u_1
        inst✝³ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝² : CommRing P
        inst✝¹ : Algebra R P
        inst✝ : IsLocalization S P
        x y : P
        I : FractionalIdeal S P
        h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y
        ⊢ ∀ (x_1 y : P), (Exists fun y' => And (Membership.mem I y') (Eq x_1 (HMul.hMu …
      -/
    · rintro _ _ ⟨y, hy, rfl⟩ ⟨y', hy', rfl⟩
      /-
        case mp.refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommRing R
        S : Submonoid R
        P : Type u_2
        inst✝² : CommRing P
        inst✝¹ : Algebra R P
        inst✝ : IsLocalization S P
        x y✝ : P
        I : FractionalIdeal S P
        h : Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) y✝
        y : P
        hy : Membership.mem I y
        y' : P
        hy' : Membership.mem I y'
        ⊢ Exists fun y'_1 => And (Membership.mem I y'_1) (Eq (HAdd.hAdd (HMul.hMul x y …
      -/
      exact ⟨y + y', Submodule.add_mem (I : Submodule R P) hy hy', (mul_add _ _ _).symm⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x y : P
      I : FractionalIdeal S P
      ⊢ (Exists fun y' => And (Membership.mem I y') (Eq y (HMul.hMul x y'))) → Membe …
    -/
  · rintro ⟨y', hy', rfl⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I : FractionalIdeal S P
      y' : P
      hy' : Membership.mem I y'
      ⊢ Membership.mem (HMul.hMul (FractionalIdeal.spanSingleton S x) I) (HMul.hMul  …
    -/
    exact mul_mem_mul ((mem_spanSingleton S).mpr ⟨1, one_smul _ _⟩) hy'
    /-
      🎉 no goals
    -/


theorem mk'_mul_coeIdeal_eq_coeIdeal {I J : Ideal R₁} {x y : R₁} (hy : y ∈ R₁⁰) :
    spanSingleton R₁⁰ (IsLocalization.mk' K x ⟨y, hy⟩) * I = (J : FractionalIdeal R₁⁰ K) ↔
      Ideal.span {x} * I = Ideal.span {y} * J := by
  have :
    spanSingleton R₁⁰ (IsLocalization.mk' _ (1 : R₁) ⟨y, hy⟩) *
        spanSingleton R₁⁰ (algebraMap R₁ K y) =
      1 := by
    rw [spanSingleton_mul_spanSingleton, mul_comm, ← IsLocalization.mk'_eq_mul_mk'_one,
      IsLocalization.mk'_self, spanSingleton_one]
  /-
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : Ideal R₁
    x y : R₁
    hy : Membership.mem (nonZeroDivisors R₁) y
    this : Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLo …
    ⊢ Iff (Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLo …
  -/
  let y' : (FractionalIdeal R₁⁰ K)ˣ := Units.mkOfMulEqOne _ _ this
  /-
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : Ideal R₁
    x y : R₁
    hy : Membership.mem (nonZeroDivisors R₁) y
    this : Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLo …
    y' : Units (FractionalIdeal (nonZeroDivisors R₁) K) := Units.mkOfMulEqOne (Fra …
    ⊢ Iff (Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLo …
  -/
  have coe_y' : ↑y' = spanSingleton R₁⁰ (IsLocalization.mk' K (1 : R₁) ⟨y, hy⟩) := rfl
  /-
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : Ideal R₁
    x y : R₁
    hy : Membership.mem (nonZeroDivisors R₁) y
    this : Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLo …
    y' : Units (FractionalIdeal (nonZeroDivisors R₁) K) := Units.mkOfMulEqOne (Fra …
    coe_y' : Eq (↑y') (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLocal …
    ⊢ Iff (Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (IsLo …
  -/
  refine Iff.trans ?_ (y'.mul_right_inj.trans coeIdeal_inj)
  rw [coe_y', coeIdeal_mul, coeIdeal_span_singleton, coeIdeal_mul, coeIdeal_span_singleton, ←
    mul_assoc, spanSingleton_mul_spanSingleton, ← mul_assoc, spanSingleton_mul_spanSingleton,
    mul_comm (mk' _ _ _), ← IsLocalization.mk'_eq_mul_mk'_one, mul_comm (mk' _ _ _), ←
    IsLocalization.mk'_eq_mul_mk'_one, IsLocalization.mk'_self, spanSingleton_one, one_mul]


theorem spanSingleton_mul_coeIdeal_eq_coeIdeal {I J : Ideal R₁} {z : K} :
    spanSingleton R₁⁰ z * (I : FractionalIdeal R₁⁰ K) = J ↔
      Ideal.span {((IsLocalization.sec R₁⁰ z).1 : R₁)} * I =
        Ideal.span {((IsLocalization.sec R₁⁰ z).2 : R₁)} * J := by
  rw [← mk'_mul_coeIdeal_eq_coeIdeal K (IsLocalization.sec R₁⁰ z).2.prop,
    IsLocalization.mk'_sec K z]


theorem one_div_spanSingleton (x : K) : 1 / spanSingleton R₁⁰ x = spanSingleton R₁⁰ x⁻¹ := by
  classical
  exact if h : x = 0 then by simp [h] else (eq_one_div_of_mul_eq_one_right _ _ (by simp [h])).symm


@[simp]
theorem div_spanSingleton (J : FractionalIdeal R₁⁰ K) (d : K) :
    J / spanSingleton R₁⁰ d = spanSingleton R₁⁰ d⁻¹ * J := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    J : FractionalIdeal (nonZeroDivisors R₁) K
    d : K
    ⊢ Eq (HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d)) (HMu …
  -/
  rw [← one_div_spanSingleton]
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    J : FractionalIdeal (nonZeroDivisors R₁) K
    d : K
    ⊢ Eq (HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d)) (HMu …
  -/
  by_cases hd : d = 0
    /-
      case pos
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Eq d 0
      ⊢ Eq (HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d)) (HMu …
    -/
  · simp only [hd, spanSingleton_zero, div_zero, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    J : FractionalIdeal (nonZeroDivisors R₁) K
    d : K
    hd : Not (Eq d 0)
    ⊢ Eq (HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d)) (HMu …
  -/
  have h_spand : spanSingleton R₁⁰ d ≠ 0 := mt spanSingleton_eq_zero_iff.mp hd
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    J : FractionalIdeal (nonZeroDivisors R₁) K
    d : K
    hd : Not (Eq d 0)
    h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
    ⊢ Eq (HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d)) (HMu …
  -/
  apply le_antisymm
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      ⊢ LE.le (HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d)) ( …
    -/
  · intro x hx
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      x : K
      hx : Membership.mem ((fun a => ↑a) (HDiv.hDiv J (FractionalIdeal.spanSingleton …
      ⊢ Membership.mem ((fun a => ↑a) (HMul.hMul (HDiv.hDiv 1 (FractionalIdeal.spanS …
    -/
    dsimp only [val_eq_coe] at hx ⊢ -- Porting note: get rid of the partially applied `coe`s
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      x : K
      hx : Membership.mem (↑(HDiv.hDiv J (FractionalIdeal.spanSingleton (nonZeroDivi …
      ⊢ Membership.mem (↑(HMul.hMul (HDiv.hDiv 1 (FractionalIdeal.spanSingleton (non …
    -/
    rw [coe_div h_spand, Submodule.mem_div_iff_forall_mul_mem] at hx
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      x : K
      hx : ∀ (y : K), Membership.mem (↑(FractionalIdeal.spanSingleton (nonZeroDiviso …
      ⊢ Membership.mem (↑(HMul.hMul (HDiv.hDiv 1 (FractionalIdeal.spanSingleton (non …
    -/
    specialize hx d (mem_spanSingleton_self R₁⁰ d)
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      x : K
      hx : Membership.mem (↑J) (HMul.hMul x d)
      ⊢ Membership.mem (↑(HMul.hMul (HDiv.hDiv 1 (FractionalIdeal.spanSingleton (non …
    -/
    have h_xd : x = d⁻¹ * (x * d) := by field_simp
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      x : K
      hx : Membership.mem (↑J) (HMul.hMul x d)
      h_xd : Eq x (HMul.hMul (Inv.inv d) (HMul.hMul x d))
      ⊢ Membership.mem (↑(HMul.hMul (HDiv.hDiv 1 (FractionalIdeal.spanSingleton (non …
    -/
    rw [coe_mul, one_div_spanSingleton, h_xd]
    /-
      case neg.a
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      J : FractionalIdeal (nonZeroDivisors R₁) K
      d : K
      hd : Not (Eq d 0)
      h_spand : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) d) 0
      x : K
      hx : Membership.mem (↑J) (HMul.hMul x d)
      h_xd : Eq x (HMul.hMul (Inv.inv d) (HMul.hMul x d))
      ⊢ Membership.mem (HMul.hMul ↑(FractionalIdeal.spanSingleton (nonZeroDivisors R …
    -/
    exact Submodule.mul_mem_mul (mem_spanSingleton_self R₁⁰ _) hx
    /-
      🎉 no goals
    -/
  · rw [le_div_iff_mul_le h_spand, mul_assoc, mul_left_comm, one_div_spanSingleton,
      spanSingleton_mul_spanSingleton, inv_mul_cancel₀ hd, spanSingleton_one, mul_one]


theorem exists_eq_spanSingleton_mul (I : FractionalIdeal R₁⁰ K) :
    ∃ (a : R₁) (aI : Ideal R₁), a ≠ 0 ∧ I = spanSingleton R₁⁰ (algebraMap R₁ K a)⁻¹ * aI := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    ⊢ Exists fun a => Exists fun aI => And (Ne a 0) (Eq I (HMul.hMul (FractionalId …
  -/
  obtain ⟨a_inv, nonzero, ha⟩ := I.isFractional
  /-
    case intro.intro
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    a_inv : R₁
    nonzero : Membership.mem (nonZeroDivisors R₁) a_inv
    ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
    ⊢ Exists fun a => Exists fun aI => And (Ne a 0) (Eq I (HMul.hMul (FractionalId …
  -/
  have nonzero := mem_nonZeroDivisors_iff_ne_zero.mp nonzero
  have map_a_nonzero : algebraMap R₁ K a_inv ≠ 0 :=
    mt IsFractionRing.to_map_eq_zero_iff.mp nonzero
  refine
    ⟨a_inv,
      Submodule.comap (Algebra.linearMap R₁ K) ↑(spanSingleton R₁⁰ (algebraMap R₁ K a_inv) * I),
      nonzero, ext fun x => Iff.trans ⟨?_, ?_⟩ mem_singleton_mul.symm⟩
    /-
      case intro.intro.refine_1
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x : K
      ⊢ Membership.mem I x → Exists fun y' => And (Membership.mem (↑(Submodule.comap …
    -/
  · intro hx
    /-
      case intro.intro.refine_1
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x : K
      hx : Membership.mem I x
      ⊢ Exists fun y' => And (Membership.mem (↑(Submodule.comap (Algebra.linearMap R …
    -/
    obtain ⟨x', hx'⟩ := ha x hx
    /-
      case intro.intro.refine_1.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x : K
      hx : Membership.mem I x
      x' : R₁
      hx' : Eq ((algebraMap R₁ K) x') (HSMul.hSMul a_inv x)
      ⊢ Exists fun y' => And (Membership.mem (↑(Submodule.comap (Algebra.linearMap R …
    -/
    rw [Algebra.smul_def] at hx'
    /-
      case intro.intro.refine_1.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x : K
      hx : Membership.mem I x
      x' : R₁
      hx' : Eq ((algebraMap R₁ K) x') (HMul.hMul ((algebraMap R₁ K) a_inv) x)
      ⊢ Exists fun y' => And (Membership.mem (↑(Submodule.comap (Algebra.linearMap R …
    -/
    refine ⟨algebraMap R₁ K x', (mem_coeIdeal _).mpr ⟨x', mem_singleton_mul.mpr ?_, rfl⟩, ?_⟩
      /-
        case intro.intro.refine_1.intro.refine_1
        R₁ : Type u_3
        inst✝⁴ : CommRing R₁
        K : Type u_4
        inst✝³ : Field K
        inst✝² : Algebra R₁ K
        inst✝¹ : IsFractionRing R₁ K
        inst✝ : IsDomain R₁
        I : FractionalIdeal (nonZeroDivisors R₁) K
        a_inv : R₁
        nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
        ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
        nonzero : Ne a_inv 0
        map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
        x : K
        hx : Membership.mem I x
        x' : R₁
        hx' : Eq ((algebraMap R₁ K) x') (HMul.hMul ((algebraMap R₁ K) a_inv) x)
        ⊢ Exists fun y' => And (Membership.mem I y') (Eq ((Algebra.linearMap R₁ K) x') …
      -/
    · exact ⟨x, hx, hx'⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_1.intro.refine_2
        R₁ : Type u_3
        inst✝⁴ : CommRing R₁
        K : Type u_4
        inst✝³ : Field K
        inst✝² : Algebra R₁ K
        inst✝¹ : IsFractionRing R₁ K
        inst✝ : IsDomain R₁
        I : FractionalIdeal (nonZeroDivisors R₁) K
        a_inv : R₁
        nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
        ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
        nonzero : Ne a_inv 0
        map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
        x : K
        hx : Membership.mem I x
        x' : R₁
        hx' : Eq ((algebraMap R₁ K) x') (HMul.hMul ((algebraMap R₁ K) a_inv) x)
        ⊢ Eq x (HMul.hMul (Inv.inv ((algebraMap R₁ K) a_inv)) ((algebraMap R₁ K) x'))
      -/
    · rw [hx', ← mul_assoc, inv_mul_cancel₀ map_a_nonzero, one_mul]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_2
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x : K
      ⊢ (Exists fun y' => And (Membership.mem (↑(Submodule.comap (Algebra.linearMap  …
    -/
  · rintro ⟨y, hy, rfl⟩
    /-
      case intro.intro.refine_2.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      y : K
      hy : Membership.mem (↑(Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (F …
      ⊢ Membership.mem I (HMul.hMul (Inv.inv ((algebraMap R₁ K) a_inv)) y)
    -/
    obtain ⟨x', hx', rfl⟩ := (mem_coeIdeal _).mp hy
    /-
      case intro.intro.refine_2.intro.intro.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x' : R₁
      hx' : Membership.mem (Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (Fr …
      hy : Membership.mem (↑(Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (F …
      ⊢ Membership.mem I (HMul.hMul (Inv.inv ((algebraMap R₁ K) a_inv)) ((algebraMap …
    -/
    obtain ⟨y', hy', hx'⟩ := mem_singleton_mul.mp hx'
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x' : R₁
      hx'✝ : Membership.mem (Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (F …
      hy : Membership.mem (↑(Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (F …
      y' : K
      hy' : Membership.mem I y'
      hx' : Eq ((Algebra.linearMap R₁ K) x') (HMul.hMul ((algebraMap R₁ K) a_inv) y')
      ⊢ Membership.mem I (HMul.hMul (Inv.inv ((algebraMap R₁ K) a_inv)) ((algebraMap …
    -/
    rw [Algebra.linearMap_apply] at hx'
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.intro
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      a_inv : R₁
      nonzero✝ : Membership.mem (nonZeroDivisors R₁) a_inv
      ha : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger R₁ (HSMul.hSM …
      nonzero : Ne a_inv 0
      map_a_nonzero : Ne ((algebraMap R₁ K) a_inv) 0
      x' : R₁
      hx'✝ : Membership.mem (Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (F …
      hy : Membership.mem (↑(Submodule.comap (Algebra.linearMap R₁ K) ↑(HMul.hMul (F …
      y' : K
      hy' : Membership.mem I y'
      hx' : Eq ((algebraMap R₁ K) x') (HMul.hMul ((algebraMap R₁ K) a_inv) y')
      ⊢ Membership.mem I (HMul.hMul (Inv.inv ((algebraMap R₁ K) a_inv)) ((algebraMap …
    -/
    rwa [hx', ← mul_assoc, inv_mul_cancel₀ map_a_nonzero, one_mul]
    /-
      🎉 no goals
    -/



/-- If `I` is a nonzero fractional ideal, `a ∈ R`, and `J` is an ideal of `R` such that
`I = a⁻¹J`, then `J` is nonzero. -/
theorem ideal_factor_ne_zero {R} [CommRing R] {K : Type*} [Field K] [Algebra R K]
    [IsFractionRing R K] {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) {a : R} {J : Ideal R}
    (haJ : I = spanSingleton R⁰ ((algebraMap R K) a)⁻¹ * ↑J) : J ≠ 0 := fun h ↦ by
  /-
    R : Type u_6
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    h : Eq J 0
    ⊢ False
  -/
  rw [h, Ideal.zero_eq_bot, coeIdeal_bot, MulZeroClass.mul_zero] at haJ
  /-
    R : Type u_6
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I 0
    h : Eq J 0
    ⊢ False
  -/
  exact hI haJ
  /-
    🎉 no goals
  -/


/-- If `I` is a nonzero fractional ideal, `a ∈ R`, and `J` is an ideal of `R` such that
`I = a⁻¹J`, then `a` is nonzero. -/
theorem constant_factor_ne_zero {R} [CommRing R] {K : Type*} [Field K] [Algebra R K]
    [IsFractionRing R K] {I : FractionalIdeal R⁰ K} (hI : I ≠ 0) {a : R} {J : Ideal R}
    (haJ : I = spanSingleton R⁰ ((algebraMap R K) a)⁻¹ * ↑J) :
    (Ideal.span {a} : Ideal R) ≠ 0 := fun h ↦ by
  /-
    R : Type u_6
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    h : Eq (Ideal.span (Singleton.singleton a)) 0
    ⊢ False
  -/
  rw [Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot] at h
  /-
    R : Type u_6
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv. …
    h : Eq a 0
    ⊢ False
  -/
  rw [h, RingHom.map_zero, inv_zero, spanSingleton_zero, MulZeroClass.zero_mul] at haJ
  /-
    R : Type u_6
    inst✝³ : CommRing R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    hI : Ne I 0
    a : R
    J : Ideal R
    haJ : Eq I 0
    h : Eq a 0
    ⊢ False
  -/
  exact hI haJ
  /-
    🎉 no goals
  -/


instance isPrincipal {R} [CommRing R] [IsDomain R] [IsPrincipalIdealRing R] [Algebra R K]
    [IsFractionRing R K] (I : FractionalIdeal R⁰ K) : (I : Submodule R K).IsPrincipal := by
  /-
    R✝ : Type u_1
    inst✝¹³ : CommRing R✝
    S : Submonoid R✝
    P : Type u_2
    inst✝¹² : CommRing P
    inst✝¹¹ : Algebra R✝ P
    R₁ : Type u_3
    inst✝¹⁰ : CommRing R₁
    K : Type u_4
    inst✝⁹ : Field K
    inst✝⁸ : Algebra R₁ K
    inst✝⁷ : IsFractionRing R₁ K
    inst✝⁶ : IsLocalization S P
    inst✝⁵ : IsDomain R₁
    R : Type u_5
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    ⊢ (↑I).IsPrincipal
  -/
  obtain ⟨a, aI, -, ha⟩ := exists_eq_spanSingleton_mul I
  /-
    case intro.intro.intro
    R✝ : Type u_1
    inst✝¹³ : CommRing R✝
    S : Submonoid R✝
    P : Type u_2
    inst✝¹² : CommRing P
    inst✝¹¹ : Algebra R✝ P
    R₁ : Type u_3
    inst✝¹⁰ : CommRing R₁
    K : Type u_4
    inst✝⁹ : Field K
    inst✝⁸ : Algebra R₁ K
    inst✝⁷ : IsFractionRing R₁ K
    inst✝⁶ : IsLocalization S P
    inst✝⁵ : IsDomain R₁
    R : Type u_5
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    a : R
    aI : Ideal R
    ha : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.i …
    ⊢ (↑I).IsPrincipal
  -/
  use (algebraMap R K a)⁻¹ * algebraMap R K (generator aI)
  suffices I = spanSingleton R⁰ ((algebraMap R K a)⁻¹ * algebraMap R K (generator aI)) by
    rw [spanSingleton] at this
    exact congr_arg Subtype.val this
  /-
    case h
    R✝ : Type u_1
    inst✝¹³ : CommRing R✝
    S : Submonoid R✝
    P : Type u_2
    inst✝¹² : CommRing P
    inst✝¹¹ : Algebra R✝ P
    R₁ : Type u_3
    inst✝¹⁰ : CommRing R₁
    K : Type u_4
    inst✝⁹ : Field K
    inst✝⁸ : Algebra R₁ K
    inst✝⁷ : IsFractionRing R₁ K
    inst✝⁶ : IsLocalization S P
    inst✝⁵ : IsDomain R₁
    R : Type u_5
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    I : FractionalIdeal (nonZeroDivisors R) K
    a : R
    aI : Ideal R
    ha : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R) (Inv.i …
    ⊢ Eq I (FractionalIdeal.spanSingleton (nonZeroDivisors R) (HMul.hMul (Inv.inv  …
  -/
  conv_lhs => rw [ha, ← span_singleton_generator aI]
  rw [Ideal.submodule_span_eq, coeIdeal_span_singleton (generator aI),
    spanSingleton_mul_spanSingleton]


theorem le_spanSingleton_mul_iff {x : P} {I J : FractionalIdeal S P} :
    I ≤ spanSingleton S x * J ↔ ∀ zI ∈ I, ∃ zJ ∈ J, x * zJ = zI :=
  show (∀ {zI} (_ : zI ∈ I), zI ∈ spanSingleton _ x * J) ↔ ∀ zI ∈ I, ∃ zJ ∈ J, x * zJ = zI by
    /-
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I J : FractionalIdeal S P
      ⊢ Iff (∀ {zI : P}, Membership.mem I zI → Membership.mem (HMul.hMul (Fractional …
    -/
    simp only [mem_singleton_mul, eq_comm]
    /-
      🎉 no goals
    -/


theorem spanSingleton_mul_le_iff {x : P} {I J : FractionalIdeal S P} :
    spanSingleton _ x * I ≤ J ↔ ∀ z ∈ I, x * z ∈ J := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    I J : FractionalIdeal S P
    ⊢ Iff (LE.le (HMul.hMul (FractionalIdeal.spanSingleton S x) I) J) (∀ (z : P),  …
  -/
  simp only [mul_le, mem_singleton_mul, mem_spanSingleton]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    I J : FractionalIdeal S P
    ⊢ Iff (∀ (i : P), (Exists fun z => Eq (HSMul.hSMul z x) i) → ∀ (j : P), Member …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I J : FractionalIdeal S P
      ⊢ (∀ (i : P), (Exists fun z => Eq (HSMul.hSMul z x) i) → ∀ (j : P), Membership …
    -/
  · intro h zI hzI
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I J : FractionalIdeal S P
      h : ∀ (i : P), (Exists fun z => Eq (HSMul.hSMul z x) i) → ∀ (j : P), Membershi …
      zI : P
      hzI : Membership.mem I zI
      ⊢ Membership.mem J (HMul.hMul x zI)
    -/
    exact h x ⟨1, one_smul _ _⟩ zI hzI
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I J : FractionalIdeal S P
      ⊢ (∀ (z : P), Membership.mem I z → Membership.mem J (HMul.hMul x z)) → ∀ (i :  …
    -/
  · rintro h _ ⟨z, rfl⟩ zI hzI
    /-
      case mpr.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I J : FractionalIdeal S P
      h : ∀ (z : P), Membership.mem I z → Membership.mem J (HMul.hMul x z)
      z : R
      zI : P
      hzI : Membership.mem I zI
      ⊢ Membership.mem J (HMul.hMul (HSMul.hSMul z x) zI)
    -/
    rw [Algebra.smul_mul_assoc]
    /-
      case mpr.intro
      R : Type u_1
      inst✝³ : CommRing R
      S : Submonoid R
      P : Type u_2
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization S P
      x : P
      I J : FractionalIdeal S P
      h : ∀ (z : P), Membership.mem I z → Membership.mem J (HMul.hMul x z)
      z : R
      zI : P
      hzI : Membership.mem I zI
      ⊢ Membership.mem J (HSMul.hSMul z (HMul.hMul x zI))
    -/
    exact Submodule.smul_mem J.1 _ (h zI hzI)
    /-
      🎉 no goals
    -/


theorem eq_spanSingleton_mul {x : P} {I J : FractionalIdeal S P} :
    I = spanSingleton _ x * J ↔ (∀ zI ∈ I, ∃ zJ ∈ J, x * zJ = zI) ∧ ∀ z ∈ J, x * z ∈ I := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    x : P
    I J : FractionalIdeal S P
    ⊢ Iff (Eq I (HMul.hMul (FractionalIdeal.spanSingleton S x) J)) (And (∀ (zI : P …
  -/
  simp only [le_antisymm_iff, le_spanSingleton_mul_iff, spanSingleton_mul_le_iff]
  /-
    🎉 no goals
  -/


theorem num_le (I : FractionalIdeal S P) :
    (I.num : FractionalIdeal S P) ≤ I := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ LE.le (↑I.num) I
  -/
  rw [← I.den_mul_self_eq_num', spanSingleton_mul_le_iff]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    ⊢ ∀ (z : P), Membership.mem I z → Membership.mem I (HMul.hMul ((algebraMap R P …
  -/
  intro _ h
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    z✝ : P
    h : Membership.mem I z✝
    ⊢ Membership.mem I (HMul.hMul ((algebraMap R P) ↑I.den) z✝)
  -/
  rw [← Algebra.smul_def]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Submonoid R
    P : Type u_2
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization S P
    I : FractionalIdeal S P
    z✝ : P
    h : Membership.mem I z✝
    ⊢ Membership.mem I (HSMul.hSMul (↑I.den) z✝)
  -/
  exact Submodule.smul_mem _ _ h
  /-
    🎉 no goals
  -/


theorem isNoetherian_zero : IsNoetherian R₁ (0 : FractionalIdeal R₁⁰ K) :=
  isNoetherian_submodule.mpr fun I (hI : I ≤ (0 : FractionalIdeal R₁⁰ K)) => by
    /-
      R₁ : Type u_3
      inst✝² : CommRing R₁
      K : Type u_4
      inst✝¹ : Field K
      inst✝ : Algebra R₁ K
      I : Submodule R₁ K
      hI : LE.le I ↑0
      ⊢ I.FG
    -/
    rw [coe_zero, le_bot_iff] at hI
    /-
      R₁ : Type u_3
      inst✝² : CommRing R₁
      K : Type u_4
      inst✝¹ : Field K
      inst✝ : Algebra R₁ K
      I : Submodule R₁ K
      hI : Eq I Bot.bot
      ⊢ I.FG
    -/
    rw [hI]
    /-
      R₁ : Type u_3
      inst✝² : CommRing R₁
      K : Type u_4
      inst✝¹ : Field K
      inst✝ : Algebra R₁ K
      I : Submodule R₁ K
      hI : Eq I Bot.bot
      ⊢ Bot.bot.FG
    -/
    exact fg_bot
    /-
      🎉 no goals
    -/


theorem isNoetherian_iff {I : FractionalIdeal R₁⁰ K} :
    IsNoetherian R₁ I ↔ ∀ J ≤ I, (J : Submodule R₁ K).FG :=
  isNoetherian_submodule.trans ⟨fun h _ hJ => h _ hJ, fun h J hJ => h ⟨J, isFractional_of_le hJ⟩ hJ⟩


theorem isNoetherian_coeIdeal [IsNoetherianRing R₁] (I : Ideal R₁) :
    IsNoetherian R₁ (I : FractionalIdeal R₁⁰ K) := by
  /-
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsNoetherianRing R₁
    I : Ideal R₁
    ⊢ IsNoetherian R₁ (Subtype fun x => Membership.mem (↑↑I) x)
  -/
  rw [isNoetherian_iff]
  /-
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsNoetherianRing R₁
    I : Ideal R₁
    ⊢ ∀ (J : FractionalIdeal (nonZeroDivisors R₁) K), LE.le J ↑I → (↑J).FG
  -/
  intro J hJ
  /-
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsNoetherianRing R₁
    I : Ideal R₁
    J : FractionalIdeal (nonZeroDivisors R₁) K
    hJ : LE.le J ↑I
    ⊢ (↑J).FG
  -/
  obtain ⟨J, rfl⟩ := le_one_iff_exists_coeIdeal.mp (le_trans hJ coeIdeal_le_one)
  /-
    case intro
    R₁ : Type u_3
    inst✝³ : CommRing R₁
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra R₁ K
    inst✝ : IsNoetherianRing R₁
    I J : Ideal R₁
    hJ : LE.le ↑J ↑I
    ⊢ (↑↑J).FG
  -/
  exact (IsNoetherian.noetherian J).map _
  /-
    🎉 no goals
  -/


theorem isNoetherian_spanSingleton_inv_to_map_mul (x : R₁) {I : FractionalIdeal R₁⁰ K}
    (hI : IsNoetherian R₁ I) :
    IsNoetherian R₁ (spanSingleton R₁⁰ (algebraMap R₁ K x)⁻¹ * I : FractionalIdeal R₁⁰ K) := by
  /-
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : IsNoetherian R₁ (Subtype fun x => Membership.mem (↑I) x)
    ⊢ IsNoetherian R₁ (Subtype fun x_1 => Membership.mem (↑(HMul.hMul (FractionalI …
  -/
  by_cases hx : x = 0
    /-
      case pos
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      x : R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : IsNoetherian R₁ (Subtype fun x => Membership.mem (↑I) x)
      hx : Eq x 0
      ⊢ IsNoetherian R₁ (Subtype fun x_1 => Membership.mem (↑(HMul.hMul (FractionalI …
    -/
  · rw [hx, RingHom.map_zero, inv_zero, spanSingleton_zero, zero_mul]
    /-
      case pos
      R₁ : Type u_3
      inst✝⁴ : CommRing R₁
      K : Type u_4
      inst✝³ : Field K
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      inst✝ : IsDomain R₁
      x : R₁
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : IsNoetherian R₁ (Subtype fun x => Membership.mem (↑I) x)
      hx : Eq x 0
      ⊢ IsNoetherian R₁ (Subtype fun x => Membership.mem (↑0) x)
    -/
    exact isNoetherian_zero
    /-
      🎉 no goals
    -/
  have h_gx : algebraMap R₁ K x ≠ 0 :=
    mt ((injective_iff_map_eq_zero (algebraMap R₁ K)).mp (IsFractionRing.injective _ _) x) hx
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : IsNoetherian R₁ (Subtype fun x => Membership.mem (↑I) x)
    hx : Not (Eq x 0)
    h_gx : Ne ((algebraMap R₁ K) x) 0
    ⊢ IsNoetherian R₁ (Subtype fun x_1 => Membership.mem (↑(HMul.hMul (FractionalI …
  -/
  have h_spanx : spanSingleton R₁⁰ (algebraMap R₁ K x) ≠ 0 := spanSingleton_ne_zero_iff.mpr h_gx
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : IsNoetherian R₁ (Subtype fun x => Membership.mem (↑I) x)
    hx : Not (Eq x 0)
    h_gx : Ne ((algebraMap R₁ K) x) 0
    h_spanx : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) ((algebraMap  …
    ⊢ IsNoetherian R₁ (Subtype fun x_1 => Membership.mem (↑(HMul.hMul (FractionalI …
  -/
  rw [isNoetherian_iff] at hI ⊢
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : ∀ (J : FractionalIdeal (nonZeroDivisors R₁) K), LE.le J I → (↑J).FG
    hx : Not (Eq x 0)
    h_gx : Ne ((algebraMap R₁ K) x) 0
    h_spanx : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) ((algebraMap  …
    ⊢ ∀ (J : FractionalIdeal (nonZeroDivisors R₁) K), LE.le J (HMul.hMul (Fraction …
  -/
  intro J hJ
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : ∀ (J : FractionalIdeal (nonZeroDivisors R₁) K), LE.le J I → (↑J).FG
    hx : Not (Eq x 0)
    h_gx : Ne ((algebraMap R₁ K) x) 0
    h_spanx : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) ((algebraMap  …
    J : FractionalIdeal (nonZeroDivisors R₁) K
    hJ : LE.le J (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (I …
    ⊢ (↑J).FG
  -/
  rw [← div_spanSingleton, le_div_iff_mul_le h_spanx] at hJ
  /-
    case neg
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : ∀ (J : FractionalIdeal (nonZeroDivisors R₁) K), LE.le J I → (↑J).FG
    hx : Not (Eq x 0)
    h_gx : Ne ((algebraMap R₁ K) x) 0
    h_spanx : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) ((algebraMap  …
    J : FractionalIdeal (nonZeroDivisors R₁) K
    hJ : LE.le (HMul.hMul J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (( …
    ⊢ (↑J).FG
  -/
  obtain ⟨s, hs⟩ := hI _ hJ
  /-
    case neg.intro
    R₁ : Type u_3
    inst✝⁴ : CommRing R₁
    K : Type u_4
    inst✝³ : Field K
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    inst✝ : IsDomain R₁
    x : R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : ∀ (J : FractionalIdeal (nonZeroDivisors R₁) K), LE.le J I → (↑J).FG
    hx : Not (Eq x 0)
    h_gx : Ne ((algebraMap R₁ K) x) 0
    h_spanx : Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) ((algebraMap  …
    J : FractionalIdeal (nonZeroDivisors R₁) K
    hJ : LE.le (HMul.hMul J (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (( …
    s : Finset K
    hs : Eq (Submodule.span R₁ ↑s) ↑(HMul.hMul J (FractionalIdeal.spanSingleton (n …
    ⊢ (↑J).FG
  -/
  use s * {(algebraMap R₁ K x)⁻¹}
  rw [Finset.coe_mul, Finset.coe_singleton, ← span_mul_span, hs, ← coe_spanSingleton R₁⁰, ←
    coe_mul, mul_assoc, spanSingleton_mul_spanSingleton, mul_inv_cancel₀ h_gx, spanSingleton_one,
    mul_one]


/-- Every fractional ideal of a noetherian integral domain is noetherian. -/
theorem isNoetherian [IsNoetherianRing R₁] (I : FractionalIdeal R₁⁰ K) : IsNoetherian R₁ I := by
  /-
    R₁ : Type u_3
    inst✝⁵ : CommRing R₁
    K : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Algebra R₁ K
    inst✝² : IsFractionRing R₁ K
    inst✝¹ : IsDomain R₁
    inst✝ : IsNoetherianRing R₁
    I : FractionalIdeal (nonZeroDivisors R₁) K
    ⊢ IsNoetherian R₁ (Subtype fun x => Membership.mem (↑I) x)
  -/
  obtain ⟨d, J, _, rfl⟩ := exists_eq_spanSingleton_mul I
  /-
    case intro.intro.intro
    R₁ : Type u_3
    inst✝⁵ : CommRing R₁
    K : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Algebra R₁ K
    inst✝² : IsFractionRing R₁ K
    inst✝¹ : IsDomain R₁
    inst✝ : IsNoetherianRing R₁
    d : R₁
    J : Ideal R₁
    left✝ : Ne d 0
    ⊢ IsNoetherian R₁ (Subtype fun x => Membership.mem (↑(HMul.hMul (FractionalIde …
  -/
  apply isNoetherian_spanSingleton_inv_to_map_mul
  /-
    case intro.intro.intro.hI
    R₁ : Type u_3
    inst✝⁵ : CommRing R₁
    K : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Algebra R₁ K
    inst✝² : IsFractionRing R₁ K
    inst✝¹ : IsDomain R₁
    inst✝ : IsNoetherianRing R₁
    d : R₁
    J : Ideal R₁
    left✝ : Ne d 0
    ⊢ IsNoetherian R₁ (Subtype fun x => Membership.mem (↑↑J) x)
  -/
  apply isNoetherian_coeIdeal
  /-
    🎉 no goals
  -/


/-- `A[x]` is a fractional ideal for every integral `x`. -/
theorem isFractional_adjoin_integral (hx : IsIntegral R x) :
    IsFractional S (Subalgebra.toSubmodule (Algebra.adjoin R ({x} : Set P))) :=
  isFractional_of_fg hx.fg_adjoin_singleton


/-- `FractionalIdeal.adjoinIntegral (S : Submonoid R) x hx` is `R[x]` as a fractional ideal,
where `hx` is a proof that `x : P` is integral over `R`. -/
-- Porting note: `@[simps]` generated a `Subtype.val` coercion instead of a
-- `FractionalIdeal.coeToSubmodule` coercion
def adjoinIntegral (hx : IsIntegral R x) : FractionalIdeal S P :=
  ⟨_, isFractional_adjoin_integral S x hx⟩


@[simp]
theorem adjoinIntegral_coe (hx : IsIntegral R x) :
    (adjoinIntegral S x hx : Submodule R P) =
      (Subalgebra.toSubmodule (Algebra.adjoin R ({x} : Set P))) :=
  rfl


theorem mem_adjoinIntegral_self (hx : IsIntegral R x) : x ∈ adjoinIntegral S x hx :=
  Algebra.subset_adjoin (Set.mem_singleton x)


