/--
An `R`-module `Q` is injective if and only if every injective `R`-linear map descends to a linear
map to `Q`, i.e. in the following diagram, if `f` is injective then there is an `R`-linear map
`h : Y ⟶ Q` such that `g = h ∘ f`
  ```
  X --- f ---> Y
  |
  | g
  v
  Q
  ```
-/
@[mk_iff] class Module.Injective : Prop where
  out : ∀ ⦃X Y : Type v⦄ [AddCommGroup X] [AddCommGroup Y] [Module R X] [Module R Y]
    (f : X →ₗ[R] Y) (_ : Function.Injective f) (g : X →ₗ[R] Q),
    ∃ h : Y →ₗ[R] Q, ∀ x, h (f x) = g x


/-- An `R`-module `Q` satisfies Baer's criterion if any `R`-linear map from an `Ideal R` extends to
an `R`-linear map `R ⟶ Q`-/
def Module.Baer : Prop :=
  ∀ (I : Ideal R) (g : I →ₗ[R] Q), ∃ g' : R →ₗ[R] Q, ∀ (x : R) (mem : x ∈ I), g' x = g ⟨x, mem⟩


lemma of_equiv (e : Q ≃ₗ[R] M) (h : Module.Baer R Q) : Module.Baer R M := fun I g ↦
  have ⟨g', h'⟩ := h I (e.symm ∘ₗ g)
               /-
                 R : Type u
                 inst✝⁴ : Ring R
                 Q : Type v
                 inst✝³ : AddCommGroup Q
                 inst✝² : Module R Q
                 M : Type u_1
                 inst✝¹ : AddCommGroup M
                 inst✝ : Module R M
                 e : LinearEquiv (RingHom.id R) Q M
                 h : Module.Baer R Q
                 I : Ideal R
                 g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem I x) M
                 g' : LinearMap (RingHom.id R) R Q
                 h' : ∀ (x : R) (mem : Membership.mem I x), Eq (g' x) (((↑e.symm).comp g) ⟨x, m …
                 ⊢ ∀ (x : R) (mem : Membership.mem I x), Eq (((↑e).comp g') x) (g ⟨x, mem⟩)
               -/
  ⟨e ∘ₗ g', by simpa [LinearEquiv.eq_symm_apply] using h'⟩
               /-
                 🎉 no goals
               -/


lemma congr (e : Q ≃ₗ[R] M) : Module.Baer R Q ↔ Module.Baer R M := ⟨of_equiv e, of_equiv e.symm⟩


/-- If we view `M` as a submodule of `N` via the injective linear map `i : M ↪ N`, then a submodule
between `M` and `N` is a submodule `N'` of `N`. To prove Baer's criterion, we need to consider
pairs of `(N', f')` such that `M ≤ N' ≤ N` and `f'` extends `f`. -/
structure ExtensionOf extends LinearPMap R N Q where
  le : LinearMap.range i ≤ domain
  is_extension : ∀ m : M, f m = toLinearPMap ⟨i m, le ⟨m, rfl⟩⟩


@[ext (iff := false)]
theorem ExtensionOf.ext {a b : ExtensionOf i f} (domain_eq : a.domain = b.domain)
    (to_fun_eq :
      ∀ ⦃x : a.domain⦄ ⦃y : b.domain⦄, (x : N) = y → a.toLinearPMap x = b.toLinearPMap y) :
    a = b := by
  /-
    R : Type u
    inst✝⁶ : Ring R
    Q : Type v
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    a b : Module.Baer.ExtensionOf i f
    domain_eq : Eq a.domain b.domain
    to_fun_eq : ∀ ⦃x : Subtype fun x => Membership.mem a.domain x⦄ ⦃y : Subtype fu …
    ⊢ Eq a b
  -/
  rcases a with ⟨a, a_le, e1⟩
  /-
    case mk
    R : Type u
    inst✝⁶ : Ring R
    Q : Type v
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    b : Module.Baer.ExtensionOf i f
    a : LinearPMap R N Q
    a_le : LE.le (LinearMap.range i) a.domain
    e1 : ∀ (m : M), Eq (f m) (↑a ⟨i m, ⋯⟩)
    domain_eq : Eq { toLinearPMap := a, le := a_le, is_extension := e1 }.domain b. …
    to_fun_eq : ∀ ⦃x : Subtype fun x => Membership.mem { toLinearPMap := a, le :=  …
    ⊢ Eq { toLinearPMap := a, le := a_le, is_extension := e1 } b
  -/
  rcases b with ⟨b, b_le, e2⟩
  /-
    case mk.mk
    R : Type u
    inst✝⁶ : Ring R
    Q : Type v
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    a : LinearPMap R N Q
    a_le : LE.le (LinearMap.range i) a.domain
    e1 : ∀ (m : M), Eq (f m) (↑a ⟨i m, ⋯⟩)
    b : LinearPMap R N Q
    b_le : LE.le (LinearMap.range i) b.domain
    e2 : ∀ (m : M), Eq (f m) (↑b ⟨i m, ⋯⟩)
    domain_eq : Eq { toLinearPMap := a, le := a_le, is_extension := e1 }.domain {  …
    to_fun_eq : ∀ ⦃x : Subtype fun x => Membership.mem { toLinearPMap := a, le :=  …
    ⊢ Eq { toLinearPMap := a, le := a_le, is_extension := e1 } { toLinearPMap := b …
  -/
  congr
  /-
    case mk.mk.e_toLinearPMap
    R : Type u
    inst✝⁶ : Ring R
    Q : Type v
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    a : LinearPMap R N Q
    a_le : LE.le (LinearMap.range i) a.domain
    e1 : ∀ (m : M), Eq (f m) (↑a ⟨i m, ⋯⟩)
    b : LinearPMap R N Q
    b_le : LE.le (LinearMap.range i) b.domain
    e2 : ∀ (m : M), Eq (f m) (↑b ⟨i m, ⋯⟩)
    domain_eq : Eq { toLinearPMap := a, le := a_le, is_extension := e1 }.domain {  …
    to_fun_eq : ∀ ⦃x : Subtype fun x => Membership.mem { toLinearPMap := a, le :=  …
    ⊢ Eq a b
  -/
  exact LinearPMap.ext domain_eq to_fun_eq
  /-
    🎉 no goals
  -/


theorem ExtensionOf.ext_iff {a b : ExtensionOf i f} :
    a = b ↔ ∃ _ : a.domain = b.domain, ∀ ⦃x : a.domain⦄ ⦃y : b.domain⦄,
    (x : N) = y → a.toLinearPMap x = b.toLinearPMap y :=
  ⟨fun r => r ▸ ⟨rfl, fun _ _ h => congr_arg a.toFun <| mod_cast h⟩, fun ⟨h1, h2⟩ =>
    ExtensionOf.ext h1 h2⟩


instance : Min (ExtensionOf i f) where
  min X1 X2 :=
    { X1.toLinearPMap ⊓
        X2.toLinearPMap with
      le := fun x hx =>
        (by
          /-
            R : Type u
            inst✝⁶ : Ring R
            Q : Type v
            inst✝⁵ : AddCommGroup Q
            inst✝⁴ : Module R Q
            M : Type u_1
            N : Type u_2
            inst✝³ : AddCommGroup M
            inst✝² : AddCommGroup N
            inst✝¹ : Module R M
            inst✝ : Module R N
            i : LinearMap (RingHom.id R) M N
            f : LinearMap (RingHom.id R) M Q
            X1 X2 : Module.Baer.ExtensionOf i f
            x : N
            hx : Membership.mem (LinearMap.range i) x
            ⊢ Membership.mem (X1.eqLocus X2.toLinearPMap) x
          -/
          rcases hx with ⟨x, rfl⟩
          /-
            case intro
            R : Type u
            inst✝⁶ : Ring R
            Q : Type v
            inst✝⁵ : AddCommGroup Q
            inst✝⁴ : Module R Q
            M : Type u_1
            N : Type u_2
            inst✝³ : AddCommGroup M
            inst✝² : AddCommGroup N
            inst✝¹ : Module R M
            inst✝ : Module R N
            i : LinearMap (RingHom.id R) M N
            f : LinearMap (RingHom.id R) M Q
            X1 X2 : Module.Baer.ExtensionOf i f
            x : M
            ⊢ Membership.mem (X1.eqLocus X2.toLinearPMap) (i x)
          -/
          refine ⟨X1.le (Set.mem_range_self _), X2.le (Set.mem_range_self _), ?_⟩
          /-
            case intro
            R : Type u
            inst✝⁶ : Ring R
            Q : Type v
            inst✝⁵ : AddCommGroup Q
            inst✝⁴ : Module R Q
            M : Type u_1
            N : Type u_2
            inst✝³ : AddCommGroup M
            inst✝² : AddCommGroup N
            inst✝¹ : Module R M
            inst✝ : Module R N
            i : LinearMap (RingHom.id R) M N
            f : LinearMap (RingHom.id R) M Q
            X1 X2 : Module.Baer.ExtensionOf i f
            x : M
            ⊢ Eq (↑X1.toLinearPMap ⟨i x, ⋯⟩) (↑X2.toLinearPMap ⟨i x, ⋯⟩)
          -/
          rw [← X1.is_extension x, ← X2.is_extension x] :
          /-
            🎉 no goals
          -/
          x ∈ X1.toLinearPMap.eqLocus X2.toLinearPMap)
      is_extension := fun _ => X1.is_extension _ }


instance : SemilatticeInf (ExtensionOf i f) :=
  Function.Injective.semilatticeInf ExtensionOf.toLinearPMap
    (fun X Y h =>
                          /-
                            R : Type u
                            inst✝⁶ : Ring R
                            Q : Type v
                            inst✝⁵ : AddCommGroup Q
                            inst✝⁴ : Module R Q
                            M : Type u_1
                            N : Type u_2
                            inst✝³ : AddCommGroup M
                            inst✝² : AddCommGroup N
                            inst✝¹ : Module R M
                            inst✝ : Module R N
                            i : LinearMap (RingHom.id R) M N
                            f : LinearMap (RingHom.id R) M Q
                            X Y : Module.Baer.ExtensionOf i f
                            h : Eq X.toLinearPMap Y.toLinearPMap
                            ⊢ Eq X.domain Y.domain
                          -/
      ExtensionOf.ext (by rw [h]) fun x y h' => by
                          /-
                            🎉 no goals
                          -/
        -- Porting note: induction didn't handle dependent rw like in Lean 3
        have : {x y : N} → (h'' : x = y) → (hx : x ∈ X.toLinearPMap.domain) →
          (hy : y ∈ Y.toLinearPMap.domain) → X.toLinearPMap ⟨x,hx⟩ = Y.toLinearPMap ⟨y,hy⟩ := by
            rw [h]
            intro _ _ h _ _
            congr
        /-
          R : Type u
          inst✝⁶ : Ring R
          Q : Type v
          inst✝⁵ : AddCommGroup Q
          inst✝⁴ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝³ : AddCommGroup M
          inst✝² : AddCommGroup N
          inst✝¹ : Module R M
          inst✝ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          X Y : Module.Baer.ExtensionOf i f
          h : Eq X.toLinearPMap Y.toLinearPMap
          x : Subtype fun x => Membership.mem X.domain x
          y : Subtype fun x => Membership.mem Y.domain x
          h' : Eq ↑x ↑y
          this : ∀ {x y : N}, Eq x y → ∀ (hx : Membership.mem X.domain x) (hy : Membersh …
          ⊢ Eq (↑X.toLinearPMap x) (↑Y.toLinearPMap y)
        -/
        apply this h' _ _)
        /-
          🎉 no goals
        -/
    fun X Y =>
    LinearPMap.ext rfl fun x y h => by
      /-
        R : Type u
        inst✝⁶ : Ring R
        Q : Type v
        inst✝⁵ : AddCommGroup Q
        inst✝⁴ : Module R Q
        M : Type u_1
        N : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R M
        inst✝ : Module R N
        i : LinearMap (RingHom.id R) M N
        f : LinearMap (RingHom.id R) M Q
        X Y : Module.Baer.ExtensionOf i f
        x : Subtype fun x => Membership.mem (Min.min X Y).domain x
        y : Subtype fun x => Membership.mem (Min.min X.toLinearPMap Y.toLinearPMap).do …
        h : Eq ↑x ↑y
        ⊢ Eq (↑(Min.min X Y).toLinearPMap x) (↑(Min.min X.toLinearPMap Y.toLinearPMap) …
      -/
      congr
      /-
        case e_a
        R : Type u
        inst✝⁶ : Ring R
        Q : Type v
        inst✝⁵ : AddCommGroup Q
        inst✝⁴ : Module R Q
        M : Type u_1
        N : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R M
        inst✝ : Module R N
        i : LinearMap (RingHom.id R) M N
        f : LinearMap (RingHom.id R) M Q
        X Y : Module.Baer.ExtensionOf i f
        x : Subtype fun x => Membership.mem (Min.min X Y).domain x
        y : Subtype fun x => Membership.mem (Min.min X.toLinearPMap Y.toLinearPMap).do …
        h : Eq ↑x ↑y
        ⊢ Eq x y
      -/
      exact mod_cast h
      /-
        🎉 no goals
      -/


theorem chain_linearPMap_of_chain_extensionOf {c : Set (ExtensionOf i f)}
    (hchain : IsChain (· ≤ ·) c) :
    IsChain (· ≤ ·) <| (fun x : ExtensionOf i f => x.toLinearPMap) '' c := by
  /-
    R : Type u
    inst✝⁶ : Ring R
    Q : Type v
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    c : Set (Module.Baer.ExtensionOf i f)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    ⊢ IsChain (fun x1 x2 => LE.le x1 x2) (Set.image (fun x => x.toLinearPMap) c)
  -/
  rintro _ ⟨a, a_mem, rfl⟩ _ ⟨b, b_mem, rfl⟩ neq
  /-
    case intro.intro.intro.intro
    R : Type u
    inst✝⁶ : Ring R
    Q : Type v
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    c : Set (Module.Baer.ExtensionOf i f)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    a : Module.Baer.ExtensionOf i f
    a_mem : Membership.mem c a
    b : Module.Baer.ExtensionOf i f
    b_mem : Membership.mem c b
    neq : Ne ((fun x => x.toLinearPMap) a) ((fun x => x.toLinearPMap) b)
    ⊢ Or ((fun x1 x2 => LE.le x1 x2) ((fun x => x.toLinearPMap) a) ((fun x => x.to …
  -/
  exact hchain a_mem b_mem (ne_of_apply_ne _ neq)
  /-
    🎉 no goals
  -/


/-- The maximal element of every nonempty chain of `extension_of i f`. -/
def ExtensionOf.max {c : Set (ExtensionOf i f)} (hchain : IsChain (· ≤ ·) c)
    (hnonempty : c.Nonempty) : ExtensionOf i f :=
  { LinearPMap.sSup _
      (IsChain.directedOn <|
        chain_linearPMap_of_chain_extensionOf
          hchain) with
    le := by
      refine le_trans hnonempty.some.le <|
        (LinearPMap.le_sSup _ <|
            (Set.mem_image _ _ _).mpr ⟨hnonempty.some, hnonempty.choose_spec, rfl⟩).1
    is_extension := fun m => by
      /-
        R : Type u
        inst✝⁶ : Ring R
        Q : Type v
        inst✝⁵ : AddCommGroup Q
        inst✝⁴ : Module R Q
        M : Type u_1
        N : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R M
        inst✝ : Module R N
        i : LinearMap (RingHom.id R) M N
        f : LinearMap (RingHom.id R) M Q
        c : Set (Module.Baer.ExtensionOf i f)
        hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
        hnonempty : c.Nonempty
        m : M
        ⊢ Eq (f m) (↑__src✝ ⟨i m, ⋯⟩)
      -/
      refine Eq.trans (hnonempty.some.is_extension m) ?_
      /-
        R : Type u
        inst✝⁶ : Ring R
        Q : Type v
        inst✝⁵ : AddCommGroup Q
        inst✝⁴ : Module R Q
        M : Type u_1
        N : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R M
        inst✝ : Module R N
        i : LinearMap (RingHom.id R) M N
        f : LinearMap (RingHom.id R) M Q
        c : Set (Module.Baer.ExtensionOf i f)
        hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
        hnonempty : c.Nonempty
        m : M
        ⊢ Eq (↑hnonempty.some.toLinearPMap ⟨i m, ⋯⟩) (↑__src✝ ⟨i m, ⋯⟩)
      -/
      symm
      /-
        R : Type u
        inst✝⁶ : Ring R
        Q : Type v
        inst✝⁵ : AddCommGroup Q
        inst✝⁴ : Module R Q
        M : Type u_1
        N : Type u_2
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R M
        inst✝ : Module R N
        i : LinearMap (RingHom.id R) M N
        f : LinearMap (RingHom.id R) M Q
        c : Set (Module.Baer.ExtensionOf i f)
        hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
        hnonempty : c.Nonempty
        m : M
        ⊢ Eq (↑__src✝ ⟨i m, ⋯⟩) (↑hnonempty.some.toLinearPMap ⟨i m, ⋯⟩)
      -/
      generalize_proofs _ h1
      exact
        LinearPMap.sSup_apply (IsChain.directedOn <| chain_linearPMap_of_chain_extensionOf hchain)
          ((Set.mem_image _ _ _).mpr ⟨hnonempty.some, hnonempty.choose_spec, rfl⟩) ⟨i m, h1⟩ }


theorem ExtensionOf.le_max {c : Set (ExtensionOf i f)} (hchain : IsChain (· ≤ ·) c)
    (hnonempty : c.Nonempty) (a : ExtensionOf i f) (ha : a ∈ c) :
    a ≤ ExtensionOf.max hchain hnonempty :=
  LinearPMap.le_sSup (IsChain.directedOn <| chain_linearPMap_of_chain_extensionOf hchain) <|
    (Set.mem_image _ _ _).mpr ⟨a, ha, rfl⟩


instance ExtensionOf.inhabited : Inhabited (ExtensionOf i f) where
  default :=
    { domain := LinearMap.range i
      toFun :=
        { toFun := fun x => f x.2.choose
          map_add' := fun x y => by
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              x y : Subtype fun x => Membership.mem (LinearMap.range i) x
              ⊢ Eq ((fun x => f (Exists.choose ⋯)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f  …
            -/
            have eq1 : _ + _ = (x + y).1 := congr_arg₂ (· + ·) x.2.choose_spec y.2.choose_spec
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              x y : Subtype fun x => Membership.mem (LinearMap.range i) x
              eq1 : Eq (HAdd.hAdd (i (Exists.choose ⋯)) (i (Exists.choose ⋯))) ↑(HAdd.hAdd x …
              ⊢ Eq ((fun x => f (Exists.choose ⋯)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f  …
            -/
            rw [← map_add, ← (x + y).2.choose_spec] at eq1
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              x y : Subtype fun x => Membership.mem (LinearMap.range i) x
              eq1 : Eq (i (HAdd.hAdd (Exists.choose ⋯) (Exists.choose ⋯))) (i (Exists.choose …
              ⊢ Eq ((fun x => f (Exists.choose ⋯)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f  …
            -/
            dsimp
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              x y : Subtype fun x => Membership.mem (LinearMap.range i) x
              eq1 : Eq (i (HAdd.hAdd (Exists.choose ⋯) (Exists.choose ⋯))) (i (Exists.choose …
              ⊢ Eq (f (Exists.choose ⋯)) (HAdd.hAdd (f (Exists.choose ⋯)) (f (Exists.choose  …
            -/
            rw [← Fact.out (p := Function.Injective i) eq1, map_add]
            /-
              🎉 no goals
            -/
          map_smul' := fun r x => by
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              r : R
              x : Subtype fun x => Membership.mem (LinearMap.range i) x
              ⊢ Eq ({ toFun := fun x => f (Exists.choose ⋯), map_add' := ⋯ }.toFun (HSMul.hS …
            -/
            have eq1 : r • _ = (r • x).1 := congr_arg (r • ·) x.2.choose_spec
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              r : R
              x : Subtype fun x => Membership.mem (LinearMap.range i) x
              eq1 : Eq (HSMul.hSMul r (i (Exists.choose ⋯))) ↑(HSMul.hSMul r x)
              ⊢ Eq ({ toFun := fun x => f (Exists.choose ⋯), map_add' := ⋯ }.toFun (HSMul.hS …
            -/
            rw [← LinearMap.map_smul, ← (r • x).2.choose_spec] at eq1
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              r : R
              x : Subtype fun x => Membership.mem (LinearMap.range i) x
              eq1 : Eq (i (HSMul.hSMul r (Exists.choose ⋯))) (i (Exists.choose ⋯))
              ⊢ Eq ({ toFun := fun x => f (Exists.choose ⋯), map_add' := ⋯ }.toFun (HSMul.hS …
            -/
            dsimp
            /-
              R : Type u
              inst✝⁷ : Ring R
              Q : Type v
              inst✝⁶ : AddCommGroup Q
              inst✝⁵ : Module R Q
              M : Type u_1
              N : Type u_2
              inst✝⁴ : AddCommGroup M
              inst✝³ : AddCommGroup N
              inst✝² : Module R M
              inst✝¹ : Module R N
              i : LinearMap (RingHom.id R) M N
              f : LinearMap (RingHom.id R) M Q
              inst✝ : Fact (Function.Injective ⇑i)
              r : R
              x : Subtype fun x => Membership.mem (LinearMap.range i) x
              eq1 : Eq (i (HSMul.hSMul r (Exists.choose ⋯))) (i (Exists.choose ⋯))
              ⊢ Eq (f (Exists.choose ⋯)) (HSMul.hSMul r (f (Exists.choose ⋯)))
            -/
            rw [← Fact.out (p := Function.Injective i) eq1, LinearMap.map_smul] }
            /-
              🎉 no goals
            -/
      le := le_refl _
      is_extension := fun m => by
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          m : M
          ⊢ Eq (f m) (↑{ domain := LinearMap.range i, toFun := { toFun := fun x => f (Ex …
        -/
        simp only [LinearPMap.mk_apply, LinearMap.coe_mk]
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          m : M
          ⊢ Eq (f m) ({ toFun := fun x => f (Exists.choose ⋯), map_add' := ⋯ } ⟨i m, ⋯⟩)
        -/
        dsimp
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          m : M
          ⊢ Eq (f m) (f (Exists.choose ⋯))
        -/
        apply congrArg
        exact Fact.out (p := Function.Injective i)
          (⟨i m, ⟨_, rfl⟩⟩ : LinearMap.range i).2.choose_spec.symm }


/-- Since every nonempty chain has a maximal element, by Zorn's lemma, there is a maximal
`extension_of i f`. -/
def extensionOfMax : ExtensionOf i f :=
  (@zorn_le_nonempty (ExtensionOf i f) _ ⟨Inhabited.default⟩ fun _ hchain hnonempty =>
      ⟨ExtensionOf.max hchain hnonempty, ExtensionOf.le_max hchain hnonempty⟩).choose


theorem extensionOfMax_is_max :
    ∀ (a : ExtensionOf i f), extensionOfMax i f ≤ a → a = extensionOfMax i f :=
  fun _ ↦ (@zorn_le_nonempty (ExtensionOf i f) _ ⟨Inhabited.default⟩ fun _ hchain hnonempty =>
    ⟨ExtensionOf.max hchain hnonempty, ExtensionOf.le_max hchain hnonempty⟩).choose_spec.eq_of_ge

-- Porting note: helper function. Lean looks for an instance of `Sup (Type u)` when the
-- right hand side is substituted in directly

abbrev supExtensionOfMaxSingleton (y : N) : Submodule R N :=
  (extensionOfMax i f).domain ⊔ (Submodule.span R {y})


private theorem extensionOfMax_adjoin.aux1 {y : N} (x : supExtensionOfMaxSingleton i f y) :
    ∃ (a : (extensionOfMax i f).domain) (b : R), x.1 = a.1 + b • y := by
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    ⊢ Exists fun a => Exists fun b => Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul b y))
  -/
  have mem1 : x.1 ∈ (_ : Set _) := x.2
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    mem1 : Membership.mem ↑(Module.Baer.supExtensionOfMaxSingleton i f y) ↑x
    ⊢ Exists fun a => Exists fun b => Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul b y))
  -/
  rw [Submodule.coe_sup] at mem1
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    mem1 : Membership.mem (HAdd.hAdd ↑(Module.Baer.extensionOfMax i f).domain ↑(Su …
    ⊢ Exists fun a => Exists fun b => Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul b y))
  -/
  rcases mem1 with ⟨a, a_mem, b, b_mem : b ∈ (Submodule.span R _ : Submodule R N), eq1⟩
  /-
    case intro.intro.intro.intro
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    a : N
    a_mem : Membership.mem (↑(Module.Baer.extensionOfMax i f).domain) a
    b : N
    b_mem : Membership.mem (Submodule.span R (Singleton.singleton y)) b
    eq1 : Eq ((fun x1 x2 => HAdd.hAdd x1 x2) a b) ↑x
    ⊢ Exists fun a => Exists fun b => Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul b y))
  -/
  rw [Submodule.mem_span_singleton] at b_mem
  /-
    case intro.intro.intro.intro
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    a : N
    a_mem : Membership.mem (↑(Module.Baer.extensionOfMax i f).domain) a
    b : N
    b_mem : Exists fun a => Eq (HSMul.hSMul a y) b
    eq1 : Eq ((fun x1 x2 => HAdd.hAdd x1 x2) a b) ↑x
    ⊢ Exists fun a => Exists fun b => Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul b y))
  -/
  rcases b_mem with ⟨z, eq2⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    a : N
    a_mem : Membership.mem (↑(Module.Baer.extensionOfMax i f).domain) a
    b : N
    eq1 : Eq ((fun x1 x2 => HAdd.hAdd x1 x2) a b) ↑x
    z : R
    eq2 : Eq (HSMul.hSMul z y) b
    ⊢ Exists fun a => Exists fun b => Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul b y))
  -/
  exact ⟨⟨a, a_mem⟩, z, by rw [← eq1, ← eq2]⟩
  /-
    🎉 no goals
  -/


/-- If `x ∈ M ⊔ ⟨y⟩`, then `x = m + r • y`, `fst` pick an arbitrary such `m`. -/
def ExtensionOfMaxAdjoin.fst {y : N} (x : supExtensionOfMaxSingleton i f y) :
    (extensionOfMax i f).domain :=
  (extensionOfMax_adjoin.aux1 i x).choose


/-- If `x ∈ M ⊔ ⟨y⟩`, then `x = m + r • y`, `snd` pick an arbitrary such `r`. -/
def ExtensionOfMaxAdjoin.snd {y : N} (x : supExtensionOfMaxSingleton i f y) : R :=
  (extensionOfMax_adjoin.aux1 i x).choose_spec.choose


theorem ExtensionOfMaxAdjoin.eqn {y : N} (x : supExtensionOfMaxSingleton i f y) :
    ↑x = ↑(ExtensionOfMaxAdjoin.fst i x) + ExtensionOfMaxAdjoin.snd i x • y :=
  (extensionOfMax_adjoin.aux1 i x).choose_spec.choose_spec


/-- The ideal `I = {r | r • y ∈ N}`-/
def ExtensionOfMaxAdjoin.ideal (y : N) : Ideal R :=
  (extensionOfMax i f).domain.comap ((LinearMap.id : R →ₗ[R] R).smulRight y)


/-- A linear map `I ⟶ Q` by `x ↦ f' (x • y)` where `f'` is the maximal extension -/
def ExtensionOfMaxAdjoin.idealTo (y : N) : ExtensionOfMaxAdjoin.ideal i f y →ₗ[R] Q where
  toFun (z : { x // x ∈ ideal i f y }) := (extensionOfMax i f).toLinearPMap ⟨(↑z : R) • y, z.prop⟩
  map_add' (z1 z2 : { x // x ∈ ideal i f y }) := by
    -- Porting note: a single simp took care of the goal before reenableeta
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      y : N
      z1 z2 : Subtype fun x => Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.idea …
      ⊢ Eq ((fun z => ↑(Module.Baer.extensionOfMax i f).toLinearPMap ⟨HSMul.hSMul (↑ …
    -/
    simp_rw [← (extensionOfMax i f).toLinearPMap.map_add]
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      y : N
      z1 z2 : Subtype fun x => Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.idea …
      ⊢ Eq (↑(Module.Baer.extensionOfMax i f).toLinearPMap ⟨HSMul.hSMul (↑(HAdd.hAdd …
    -/
    congr
    /-
      case e_a.e_val
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      y : N
      z1 z2 : Subtype fun x => Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.idea …
      ⊢ Eq (HSMul.hSMul (↑(HAdd.hAdd z1 z2)) y) (HAdd.hAdd ↑⟨HSMul.hSMul (↑z1) y, ⋯⟩ …
    -/
    apply add_smul
    /-
      🎉 no goals
    -/
  map_smul' z1 (z2 : {x // x ∈ ideal i f y}) := by
    -- Porting note: a single simp took care of the goal before reenableeta
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      y : N
      z1 : R
      z2 : Subtype fun x => Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.ideal i …
      ⊢ Eq ({ toFun := fun z => ↑(Module.Baer.extensionOfMax i f).toLinearPMap ⟨HSMu …
    -/
    simp_rw [← (extensionOfMax i f).toLinearPMap.map_smul]
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      y : N
      z1 : R
      z2 : Subtype fun x => Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.ideal i …
      ⊢ Eq (↑(Module.Baer.extensionOfMax i f).toLinearPMap ⟨HSMul.hSMul (↑(HSMul.hSM …
    -/
    congr 2
    /-
      case e_a.e_val
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      y : N
      z1 : R
      z2 : Subtype fun x => Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.ideal i …
      ⊢ Eq (HSMul.hSMul (↑(HSMul.hSMul z1 z2)) y) (HSMul.hSMul ((RingHom.id R) z1) ↑ …
    -/
    apply mul_smul
    /-
      🎉 no goals
    -/


/-- Since we assumed `Q` being Baer, the linear map `x ↦ f' (x • y) : I ⟶ Q` extends to `R ⟶ Q`,
call this extended map `φ`-/
def ExtensionOfMaxAdjoin.extendIdealTo (h : Module.Baer R Q) (y : N) : R →ₗ[R] Q :=
  (h (ExtensionOfMaxAdjoin.ideal i f y) (ExtensionOfMaxAdjoin.idealTo i f y)).choose


theorem ExtensionOfMaxAdjoin.extendIdealTo_is_extension (h : Module.Baer R Q) (y : N) :
    ∀ (x : R) (mem : x ∈ ExtensionOfMaxAdjoin.ideal i f y),
      ExtensionOfMaxAdjoin.extendIdealTo i f h y x = ExtensionOfMaxAdjoin.idealTo i f y ⟨x, mem⟩ :=
  (h (ExtensionOfMaxAdjoin.ideal i f y) (ExtensionOfMaxAdjoin.idealTo i f y)).choose_spec


theorem ExtensionOfMaxAdjoin.extendIdealTo_wd' (h : Module.Baer R Q) {y : N} (r : R)
    (eq1 : r • y = 0) : ExtensionOfMaxAdjoin.extendIdealTo i f h y r = 0 := by
  have : r ∈ ideal i f y := by
    change (r • y) ∈ (extensionOfMax i f).toLinearPMap.domain
    rw [eq1]
    apply Submodule.zero_mem _
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    r : R
    eq1 : Eq (HSMul.hSMul r y) 0
    this : Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.ideal i f y) r
    ⊢ Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) 0
  -/
  rw [ExtensionOfMaxAdjoin.extendIdealTo_is_extension i f h y r this]
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    r : R
    eq1 : Eq (HSMul.hSMul r y) 0
    this : Membership.mem (Module.Baer.ExtensionOfMaxAdjoin.ideal i f y) r
    ⊢ Eq ((Module.Baer.ExtensionOfMaxAdjoin.idealTo i f y) ⟨r, this⟩) 0
  -/
  dsimp [ExtensionOfMaxAdjoin.idealTo]
  simp only [LinearMap.coe_mk, eq1, Subtype.coe_mk, ← ZeroMemClass.zero_def,
    (extensionOfMax i f).toLinearPMap.map_zero]


theorem ExtensionOfMaxAdjoin.extendIdealTo_wd (h : Module.Baer R Q) {y : N} (r r' : R)
    (eq1 : r • y = r' • y) : ExtensionOfMaxAdjoin.extendIdealTo i f h y r =
    ExtensionOfMaxAdjoin.extendIdealTo i f h y r' := by
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    r r' : R
    eq1 : Eq (HSMul.hSMul r y) (HSMul.hSMul r' y)
    ⊢ Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) ((Module.Bae …
  -/
  rw [← sub_eq_zero, ← map_sub]
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    r r' : R
    eq1 : Eq (HSMul.hSMul r y) (HSMul.hSMul r' y)
    ⊢ Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) (HSub.hSub r r' …
  -/
  convert ExtensionOfMaxAdjoin.extendIdealTo_wd' i f h (r - r') _
  /-
    case convert_2
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    r r' : R
    eq1 : Eq (HSMul.hSMul r y) (HSMul.hSMul r' y)
    ⊢ Eq (HSMul.hSMul (HSub.hSub r r') y) 0
  -/
  rw [sub_smul, sub_eq_zero, eq1]
  /-
    🎉 no goals
  -/


theorem ExtensionOfMaxAdjoin.extendIdealTo_eq (h : Module.Baer R Q) {y : N} (r : R)
    (hr : r • y ∈ (extensionOfMax i f).domain) : ExtensionOfMaxAdjoin.extendIdealTo i f h y r =
    (extensionOfMax i f).toLinearPMap ⟨r • y, hr⟩ := by
    -- Porting note: in mathlib3 `AddHom.coe_mk` was not needed
  simp only [ExtensionOfMaxAdjoin.extendIdealTo_is_extension i f h _ _ hr,
    ExtensionOfMaxAdjoin.idealTo, LinearMap.coe_mk, Subtype.coe_mk, AddHom.coe_mk]


/-- We can finally define a linear map `M ⊔ ⟨y⟩ ⟶ Q` by `x + r • y ↦ f x + φ r`
-/
def ExtensionOfMaxAdjoin.extensionToFun (h : Module.Baer R Q) {y : N} :
    supExtensionOfMaxSingleton i f y → Q := fun x =>
  (extensionOfMax i f).toLinearPMap (ExtensionOfMaxAdjoin.fst i x) +
    ExtensionOfMaxAdjoin.extendIdealTo i f h y (ExtensionOfMaxAdjoin.snd i x)


theorem ExtensionOfMaxAdjoin.extensionToFun_wd (h : Module.Baer R Q) {y : N}
    (x : supExtensionOfMaxSingleton i f y) (a : (extensionOfMax i f).domain)
    (r : R) (eq1 : ↑x = ↑a + r • y) :
    ExtensionOfMaxAdjoin.extensionToFun i f h x =
      (extensionOfMax i f).toLinearPMap a + ExtensionOfMaxAdjoin.extendIdealTo i f h y r := by
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    a : Subtype fun x => Membership.mem (Module.Baer.extensionOfMax i f).domain x
    r : R
    eq1 : Eq (↑x) (HAdd.hAdd (↑a) (HSMul.hSMul r y))
    ⊢ Eq (Module.Baer.ExtensionOfMaxAdjoin.extensionToFun i f h x) (HAdd.hAdd (↑(M …
  -/
  cases' a with a ha
  have eq2 :
    (ExtensionOfMaxAdjoin.fst i x - a : N) = (r - ExtensionOfMaxAdjoin.snd i x) • y := by
    change x = a + r • y at eq1
    rwa [ExtensionOfMaxAdjoin.eqn, ← sub_eq_zero, ← sub_sub_sub_eq, sub_eq_zero, ← sub_smul]
      at eq1
  have eq3 :=
    ExtensionOfMaxAdjoin.extendIdealTo_eq i f h (r - ExtensionOfMaxAdjoin.snd i x)
      (by rw [← eq2]; exact Submodule.sub_mem _ (ExtensionOfMaxAdjoin.fst i x).2 ha)
  /-
    case mk
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) (HSub.hSub  …
    ⊢ Eq (Module.Baer.ExtensionOfMaxAdjoin.extensionToFun i f h x) (HAdd.hAdd (↑(M …
  -/
  simp only [map_sub, sub_smul, sub_eq_iff_eq_add] at eq3
  /-
    case mk
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq (Module.Baer.ExtensionOfMaxAdjoin.extensionToFun i f h x) (HAdd.hAdd (↑(M …
  -/
  unfold ExtensionOfMaxAdjoin.extensionToFun
  /-
    case mk
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq (HAdd.hAdd (↑(Module.Baer.extensionOfMax i f).toLinearPMap (Module.Baer.E …
  -/
  rw [eq3, ← add_assoc, ← (extensionOfMax i f).toLinearPMap.map_add, AddMemClass.mk_add_mk]
  /-
    case mk
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq (HAdd.hAdd (↑(Module.Baer.extensionOfMax i f).toLinearPMap (Module.Baer.E …
  -/
  congr
  /-
    case mk.e_a.e_a
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq (Module.Baer.ExtensionOfMaxAdjoin.fst i x) ⟨HAdd.hAdd a (HSub.hSub (HSMul …
  -/
  ext
  /-
    case mk.e_a.e_a.a
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq ↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x) ↑⟨HAdd.hAdd a (HSub.hSub (HSM …
  -/
  dsimp
  /-
    case mk.e_a.e_a.a
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) (HAdd.hAdd a (HSub.hSub (HS …
  -/
  rw [Subtype.coe_mk, add_sub, ← eq1]
  /-
    case mk.e_a.e_a.a
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    x : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
    r : R
    a : N
    ha : Membership.mem (Module.Baer.extensionOfMax i f).domain a
    eq1 : Eq (↑x) (HAdd.hAdd (↑⟨a, ha⟩) (HSMul.hSMul r y))
    eq2 : Eq (HSub.hSub (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) a) (HSMul.hS …
    eq3 : Eq ((Module.Baer.ExtensionOfMaxAdjoin.extendIdealTo i f h y) r) (HAdd.hA …
    ⊢ Eq (↑(Module.Baer.ExtensionOfMaxAdjoin.fst i x)) (HSub.hSub (↑x) (HSMul.hSMu …
  -/
  exact eq_sub_of_add_eq (ExtensionOfMaxAdjoin.eqn i x).symm
  /-
    🎉 no goals
  -/


/-- The linear map `M ⊔ ⟨y⟩ ⟶ Q` by `x + r • y ↦ f x + φ r` is an extension of `f`-/
def extensionOfMaxAdjoin (h : Module.Baer R Q) (y : N) : ExtensionOf i f where
  domain := supExtensionOfMaxSingleton i f y -- (extensionOfMax i f).domain ⊔ Submodule.span R {y}
  le := le_trans (extensionOfMax i f).le le_sup_left
  toFun :=
    { toFun := ExtensionOfMaxAdjoin.extensionToFun i f h
      map_add' := fun a b => by
        have eq1 :
          ↑a + ↑b =
            ↑(ExtensionOfMaxAdjoin.fst i a + ExtensionOfMaxAdjoin.fst i b) +
              (ExtensionOfMaxAdjoin.snd i a + ExtensionOfMaxAdjoin.snd i b) • y := by
          rw [ExtensionOfMaxAdjoin.eqn, ExtensionOfMaxAdjoin.eqn, add_smul, Submodule.coe_add]
          ac_rfl
        rw [ExtensionOfMaxAdjoin.extensionToFun_wd (y := y) i f h (a + b) _ _ eq1,
          LinearPMap.map_add, map_add]
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          h : Module.Baer R Q
          y : N
          a b : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton  …
          eq1 : Eq (HAdd.hAdd ↑a ↑b) (HAdd.hAdd (↑(HAdd.hAdd (Module.Baer.ExtensionOfMax …
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑(Module.Baer.extensionOfMax i f).toLinearPMap (Mo …
        -/
        unfold ExtensionOfMaxAdjoin.extensionToFun
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          h : Module.Baer R Q
          y : N
          a b : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton  …
          eq1 : Eq (HAdd.hAdd ↑a ↑b) (HAdd.hAdd (↑(HAdd.hAdd (Module.Baer.ExtensionOfMax …
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑(Module.Baer.extensionOfMax i f).toLinearPMap (Mo …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      map_smul' := fun r a => by
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          h : Module.Baer R Q
          y : N
          r : R
          a : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
          ⊢ Eq ({ toFun := Module.Baer.ExtensionOfMaxAdjoin.extensionToFun i f h, map_ad …
        -/
        dsimp
        have eq1 :
          r • (a : N) =
            ↑(r • ExtensionOfMaxAdjoin.fst i a) + (r • ExtensionOfMaxAdjoin.snd i a) • y := by
          rw [ExtensionOfMaxAdjoin.eqn, smul_add, smul_eq_mul, mul_smul]
          rfl
        rw [ExtensionOfMaxAdjoin.extensionToFun_wd i f h (r • a :) _ _ eq1, LinearMap.map_smul,
          LinearPMap.map_smul, ← smul_add]
        /-
          R : Type u
          inst✝⁷ : Ring R
          Q : Type v
          inst✝⁶ : AddCommGroup Q
          inst✝⁵ : Module R Q
          M : Type u_1
          N : Type u_2
          inst✝⁴ : AddCommGroup M
          inst✝³ : AddCommGroup N
          inst✝² : Module R M
          inst✝¹ : Module R N
          i : LinearMap (RingHom.id R) M N
          f : LinearMap (RingHom.id R) M Q
          inst✝ : Fact (Function.Injective ⇑i)
          h : Module.Baer R Q
          y : N
          r : R
          a : Subtype fun x => Membership.mem (Module.Baer.supExtensionOfMaxSingleton i  …
          eq1 : Eq (HSMul.hSMul r ↑a) (HAdd.hAdd (↑(HSMul.hSMul r (Module.Baer.Extension …
          ⊢ Eq (HSMul.hSMul r (HAdd.hAdd (↑(Module.Baer.extensionOfMax i f).toLinearPMap …
        -/
        congr }
        /-
          🎉 no goals
        -/
  is_extension m := by
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      h : Module.Baer R Q
      y : N
      m : M
      ⊢ Eq (f m) (↑{ domain := Module.Baer.supExtensionOfMaxSingleton i f y, toFun : …
    -/
    dsimp
    rw [(extensionOfMax i f).is_extension,
      ExtensionOfMaxAdjoin.extensionToFun_wd i f h _ ⟨i m, _⟩ 0 _, map_zero, add_zero]
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      h : Module.Baer R Q
      y : N
      m : M
      ⊢ Eq (↑⟨i m, ⋯⟩) (HAdd.hAdd (↑⟨i m, ⋯⟩) (HSMul.hSMul 0 y))
    -/
    simp
    /-
      🎉 no goals
    -/


theorem extensionOfMax_le (h : Module.Baer R Q) {y : N} :
    extensionOfMax i f ≤ extensionOfMaxAdjoin i f h y :=
  ⟨le_sup_left, fun x x' EQ => by
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      h : Module.Baer R Q
      y : N
      x : Subtype fun x => Membership.mem (Module.Baer.extensionOfMax i f).domain x
      x' : Subtype fun x => Membership.mem (Module.Baer.extensionOfMaxAdjoin i f h y …
      EQ : Eq ↑x ↑x'
      ⊢ Eq (↑(Module.Baer.extensionOfMax i f).toLinearPMap x) (↑(Module.Baer.extensi …
    -/
    symm
    /-
      R : Type u
      inst✝⁷ : Ring R
      Q : Type v
      inst✝⁶ : AddCommGroup Q
      inst✝⁵ : Module R Q
      M : Type u_1
      N : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      i : LinearMap (RingHom.id R) M N
      f : LinearMap (RingHom.id R) M Q
      inst✝ : Fact (Function.Injective ⇑i)
      h : Module.Baer R Q
      y : N
      x : Subtype fun x => Membership.mem (Module.Baer.extensionOfMax i f).domain x
      x' : Subtype fun x => Membership.mem (Module.Baer.extensionOfMaxAdjoin i f h y …
      EQ : Eq ↑x ↑x'
      ⊢ Eq (↑(Module.Baer.extensionOfMaxAdjoin i f h y).toLinearPMap x') (↑(Module.B …
    -/
    change ExtensionOfMaxAdjoin.extensionToFun i f h _ = _
    rw [ExtensionOfMaxAdjoin.extensionToFun_wd i f h x' x 0 (by simp [EQ]), map_zero,
      add_zero]⟩


theorem extensionOfMax_to_submodule_eq_top (h : Module.Baer R Q) :
    (extensionOfMax i f).domain = ⊤ := by
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    ⊢ Eq (Module.Baer.extensionOfMax i f).domain Top.top
  -/
  refine Submodule.eq_top_iff'.mpr fun y => ?_
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    ⊢ Membership.mem (Module.Baer.extensionOfMax i f).domain y
  -/
  dsimp
  rw [← extensionOfMax_is_max i f _ (extensionOfMax_le i f h), extensionOfMaxAdjoin,
    Submodule.mem_sup]
  /-
    R : Type u
    inst✝⁷ : Ring R
    Q : Type v
    inst✝⁶ : AddCommGroup Q
    inst✝⁵ : Module R Q
    M : Type u_1
    N : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    i : LinearMap (RingHom.id R) M N
    f : LinearMap (RingHom.id R) M Q
    inst✝ : Fact (Function.Injective ⇑i)
    h : Module.Baer R Q
    y : N
    ⊢ Exists fun y_1 => And (Membership.mem (Module.Baer.extensionOfMax i f).domai …
  -/
  exact ⟨0, Submodule.zero_mem _, y, Submodule.mem_span_singleton_self _, zero_add _⟩
  /-
    🎉 no goals
  -/


protected theorem extension_property (h : Module.Baer R Q)
    (f : M →ₗ[R] N) (hf : Function.Injective f) (g : M →ₗ[R] Q) : ∃ h, h ∘ₗ f = g :=
  haveI : Fact (Function.Injective f) := ⟨hf⟩
  Exists.intro
    { toFun := ((extensionOfMax f g).toLinearPMap
        ⟨·, (extensionOfMax_to_submodule_eq_top f g h).symm ▸ ⟨⟩⟩)
                               /-
                                 R : Type u
                                 inst✝⁶ : Ring R
                                 Q : Type v
                                 inst✝⁵ : AddCommGroup Q
                                 inst✝⁴ : Module R Q
                                 M : Type u_1
                                 N : Type u_2
                                 inst✝³ : AddCommGroup M
                                 inst✝² : AddCommGroup N
                                 inst✝¹ : Module R M
                                 inst✝ : Module R N
                                 h : Module.Baer R Q
                                 f : LinearMap (RingHom.id R) M N
                                 hf : Function.Injective ⇑f
                                 g : LinearMap (RingHom.id R) M Q
                                 this : Fact (Function.Injective ⇑f)
                                 x y : N
                                 ⊢ Eq ((fun x => ↑(Module.Baer.extensionOfMax f g).toLinearPMap ⟨x, ⋯⟩) (HAdd.h …
                               -/
      map_add' := fun x y ↦ by rw [← LinearPMap.map_add]; congr
                                                          /-
                                                            🎉 no goals
                                                          -/
                                /-
                                  R : Type u
                                  inst✝⁶ : Ring R
                                  Q : Type v
                                  inst✝⁵ : AddCommGroup Q
                                  inst✝⁴ : Module R Q
                                  M : Type u_1
                                  N : Type u_2
                                  inst✝³ : AddCommGroup M
                                  inst✝² : AddCommGroup N
                                  inst✝¹ : Module R M
                                  inst✝ : Module R N
                                  h : Module.Baer R Q
                                  f : LinearMap (RingHom.id R) M N
                                  hf : Function.Injective ⇑f
                                  g : LinearMap (RingHom.id R) M Q
                                  this : Fact (Function.Injective ⇑f)
                                  r : R
                                  x : N
                                  ⊢ Eq ({ toFun := fun x => ↑(Module.Baer.extensionOfMax f g).toLinearPMap ⟨x, ⋯ …
                                -/
      map_smul' := fun r x ↦ by rw [← LinearPMap.map_smul]; dsimp } <|
                                                            /-
                                                              🎉 no goals
                                                            -/
    LinearMap.ext fun x ↦ ((extensionOfMax f g).is_extension x).symm


theorem extension_property_addMonoidHom (h : Module.Baer ℤ Q)
    (f : M →+ N) (hf : Function.Injective f) (g : M →+ Q) : ∃ h : N →+ Q, h.comp f = g :=
  have ⟨g', hg'⟩ := h.extension_property f.toIntLinearMap hf g.toIntLinearMap
  ⟨g', congr(LinearMap.toAddMonoidHom $hg')⟩


/-- **Baer's criterion** for injective module : a Baer module is an injective module, i.e. if every
linear map from an ideal can be extended, then the module is injective. -/
protected theorem injective (h : Module.Baer R Q) : Module.Injective R Q where
  out X Y _ _ _ _ i hi f := by
    /-
      R : Type u
      inst✝² : Ring R
      Q : Type v
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      h : Module.Baer R Q
      X Y : Type v
      x✝³ : AddCommGroup X
      x✝² : AddCommGroup Y
      x✝¹ : Module R X
      x✝ : Module R Y
      i : LinearMap (RingHom.id R) X Y
      hi : Function.Injective ⇑i
      f : LinearMap (RingHom.id R) X Q
      ⊢ Exists fun h => ∀ (x : X), Eq (h (i x)) (f x)
    -/
    obtain ⟨h, H⟩ := Module.Baer.extension_property h i hi f
    /-
      case intro
      R : Type u
      inst✝² : Ring R
      Q : Type v
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      h✝ : Module.Baer R Q
      X Y : Type v
      x✝³ : AddCommGroup X
      x✝² : AddCommGroup Y
      x✝¹ : Module R X
      x✝ : Module R Y
      i : LinearMap (RingHom.id R) X Y
      hi : Function.Injective ⇑i
      f : LinearMap (RingHom.id R) X Q
      h : LinearMap (RingHom.id R) Y Q
      H : Eq (h.comp i) f
      ⊢ Exists fun h => ∀ (x : X), Eq (h (i x)) (f x)
    -/
    exact ⟨h, DFunLike.congr_fun H⟩
    /-
      🎉 no goals
    -/


protected theorem of_injective [Small.{v} R] (inj : Module.Injective R Q) : Module.Baer R Q := by
  /-
    R : Type u
    inst✝³ : Ring R
    Q : Type v
    inst✝² : AddCommGroup Q
    inst✝¹ : Module R Q
    inst✝ : Small.{v, u} R
    inj : Module.Injective R Q
    ⊢ Module.Baer R Q
  -/
  intro I g
  /-
    R : Type u
    inst✝³ : Ring R
    Q : Type v
    inst✝² : AddCommGroup Q
    inst✝¹ : Module R Q
    inst✝ : Small.{v, u} R
    inj : Module.Injective R Q
    I : Ideal R
    g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem I x) Q
    ⊢ Exists fun g' => ∀ (x : R) (mem : Membership.mem I x), Eq (g' x) (g ⟨x, mem⟩)
  -/
  let eI := Shrink.linearEquiv I R
  /-
    R : Type u
    inst✝³ : Ring R
    Q : Type v
    inst✝² : AddCommGroup Q
    inst✝¹ : Module R Q
    inst✝ : Small.{v, u} R
    inj : Module.Injective R Q
    I : Ideal R
    g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem I x) Q
    eI : LinearEquiv (RingHom.id R) (Shrink.{v, u} (Subtype fun x => Membership.me …
    ⊢ Exists fun g' => ∀ (x : R) (mem : Membership.mem I x), Eq (g' x) (g ⟨x, mem⟩)
  -/
  let eR := Shrink.linearEquiv R R
  obtain ⟨g', hg'⟩ := Module.Injective.out (eR.symm.toLinearMap ∘ₗ I.subtype ∘ₗ eI.toLinearMap)
    (eR.symm.injective.comp <| Subtype.val_injective.comp eI.injective) (g ∘ₗ eI.toLinearMap)
  /-
    case intro
    R : Type u
    inst✝³ : Ring R
    Q : Type v
    inst✝² : AddCommGroup Q
    inst✝¹ : Module R Q
    inst✝ : Small.{v, u} R
    inj : Module.Injective R Q
    I : Ideal R
    g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem I x) Q
    eI : LinearEquiv (RingHom.id R) (Shrink.{v, u} (Subtype fun x => Membership.me …
    eR : LinearEquiv (RingHom.id R) (Shrink.{v, u} R) R := Shrink.linearEquiv R R
    g' : LinearMap (RingHom.id R) (Shrink.{v, u} R) Q
    hg' : ∀ (x : Shrink.{v, u} (Subtype fun x => Membership.mem I x)), Eq (g' (((↑ …
    ⊢ Exists fun g' => ∀ (x : R) (mem : Membership.mem I x), Eq (g' x) (g ⟨x, mem⟩)
  -/
  exact ⟨g' ∘ₗ eR.symm.toLinearMap, fun x mx ↦ by simpa [eI, eR] using hg' (equivShrink I ⟨x, mx⟩)⟩
  /-
    🎉 no goals
  -/


protected theorem iff_injective [Small.{v} R] : Module.Baer R Q ↔ Module.Injective R Q :=
  ⟨Module.Baer.injective, Module.Baer.of_injective⟩


lemma Module.ulift_injective_of_injective [Small.{v} R]
    (inj : Module.Injective R M) :
    Module.Injective R (ULift.{v'} M) := Module.Baer.injective fun I g ↦
  have ⟨g', hg'⟩ := Module.Baer.iff_injective.mpr inj I (ULift.moduleEquiv.toLinearMap ∘ₗ g)
  ⟨ULift.moduleEquiv.symm.toLinearMap ∘ₗ g', fun r hr ↦ ULift.ext _ _ <| hg' r hr⟩


lemma Module.injective_of_ulift_injective
    (inj : Module.Injective R (ULift.{v'} M)) :
    Module.Injective R M where
  out X Y _ _ _ _ f hf g :=
    let eX := ULift.moduleEquiv.{_,_,v'} (R := R) (M := X)
    have ⟨g', hg'⟩ := inj.out (ULift.moduleEquiv.{_,_,v'}.symm.toLinearMap ∘ₗ f ∘ₗ eX.toLinearMap)
          /-
            R : Type u
            inst✝² : Ring R
            M : Type v
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            inj : Module.Injective R (ULift.{v', v} M)
            X Y : Type v
            x✝³ : AddCommGroup X
            x✝² : AddCommGroup Y
            x✝¹ : Module R X
            x✝ : Module R Y
            f : LinearMap (RingHom.id R) X Y
            hf : Function.Injective ⇑f
            g : LinearMap (RingHom.id R) X M
            eX : LinearEquiv (RingHom.id R) (ULift.{v', v} X) X := ULift.moduleEquiv
            ⊢ Function.Injective ⇑((↑ULift.moduleEquiv.symm).comp (f.comp ↑eX))
          -/
      (by exact ULift.moduleEquiv.symm.injective.comp <| hf.comp eX.injective)
          /-
            🎉 no goals
          -/
      (ULift.moduleEquiv.symm.toLinearMap ∘ₗ g ∘ₗ eX.toLinearMap)
    ⟨ULift.moduleEquiv.toLinearMap ∘ₗ g' ∘ₗ ULift.moduleEquiv.symm.toLinearMap,
                 /-
                   R : Type u
                   inst✝² : Ring R
                   M : Type v
                   inst✝¹ : AddCommGroup M
                   inst✝ : Module R M
                   inj : Module.Injective R (ULift.{v', v} M)
                   X Y : Type v
                   x✝³ : AddCommGroup X
                   x✝² : AddCommGroup Y
                   x✝¹ : Module R X
                   x✝ : Module R Y
                   f : LinearMap (RingHom.id R) X Y
                   hf : Function.Injective ⇑f
                   g : LinearMap (RingHom.id R) X M
                   eX : LinearEquiv (RingHom.id R) (ULift.{v', v} X) X := ULift.moduleEquiv
                   g' : LinearMap (RingHom.id R) (ULift.{v', v} Y) (ULift.{v', v} M)
                   hg' : ∀ (x : ULift.{v', v} X), Eq (g' (((↑ULift.moduleEquiv.symm).comp (f.comp …
                   x : X
                   ⊢ Eq (((↑ULift.moduleEquiv).comp (g'.comp ↑ULift.moduleEquiv.symm)) (f x)) (g x)
                 -/
      fun x ↦ by exact congr(ULift.down $(hg' ⟨x⟩))⟩
                 /-
                   🎉 no goals
                 -/


lemma Module.injective_iff_ulift_injective :
    Module.Injective R M ↔ Module.Injective R (ULift.{v'} M) :=
  ⟨Module.ulift_injective_of_injective R,
   Module.injective_of_ulift_injective R⟩


lemma Module.Injective.extension_property
    (f : P →ₗ[R] P') (hf : Function.Injective f)
    (g : P →ₗ[R] M) : ∃ h : P' →ₗ[R] M, h ∘ₗ f = g :=
  (Module.Baer.of_injective inj).extension_property f hf g


