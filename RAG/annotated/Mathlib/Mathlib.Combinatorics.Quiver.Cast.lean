/-- Change the endpoints of an arrow using equalities. -/
def Hom.cast {u v u' v' : U} (hu : u = u') (hv : v = v') (e : u ⟶ v) : u' ⟶ v' :=
  Eq.ndrec (motive := (· ⟶ v')) (Eq.ndrec e hv) hu


theorem Hom.cast_eq_cast {u v u' v' : U} (hu : u = u') (hv : v = v') (e : u ⟶ v) :
                                   /-
                                     U : Type u_1
                                     inst✝ : Quiver U
                                     u v u' v' : U
                                     hu : Eq u u'
                                     hv : Eq v v'
                                     e : Quiver.Hom u v
                                     ⊢ Eq (Quiver.Hom u v) (Quiver.Hom u' v')
                                   -/
    e.cast hu hv = _root_.cast (by {rw [hu, hv]}) e := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    e : Quiver.Hom u v
    ⊢ Eq (Quiver.Hom.cast hu hv e) (_root_.cast ⋯ e)
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    u' v' : U
    e : Quiver.Hom u' v'
    ⊢ Eq (Quiver.Hom.cast ⋯ ⋯ e) (_root_.cast ⋯ e)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem Hom.cast_rfl_rfl {u v : U} (e : u ⟶ v) : e.cast rfl rfl = e :=
  rfl


@[simp]
theorem Hom.cast_cast {u v u' v' u'' v'' : U} (e : u ⟶ v) (hu : u = u') (hv : v = v')
    (hu' : u' = u'') (hv' : v' = v'') :
    (e.cast hu hv).cast hu' hv' = e.cast (hu.trans hu') (hv.trans hv') := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' u'' v'' : U
    e : Quiver.Hom u v
    hu : Eq u u'
    hv : Eq v v'
    hu' : Eq u' u''
    hv' : Eq v' v''
    ⊢ Eq (Quiver.Hom.cast hu' hv' (Quiver.Hom.cast hu hv e)) (Quiver.Hom.cast ⋯ ⋯ e)
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    u'' v'' : U
    e : Quiver.Hom u'' v''
    ⊢ Eq (Quiver.Hom.cast ⋯ ⋯ (Quiver.Hom.cast ⋯ ⋯ e)) (Quiver.Hom.cast ⋯ ⋯ e)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Hom.cast_heq {u v u' v' : U} (hu : u = u') (hv : v = v') (e : u ⟶ v) :
    HEq (e.cast hu hv) e := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    e : Quiver.Hom u v
    ⊢ HEq (Quiver.Hom.cast hu hv e) e
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    u' v' : U
    e : Quiver.Hom u' v'
    ⊢ HEq (Quiver.Hom.cast ⋯ ⋯ e) e
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Hom.cast_eq_iff_heq {u v u' v' : U} (hu : u = u') (hv : v = v') (e : u ⟶ v) (e' : u' ⟶ v') :
    e.cast hu hv = e' ↔ HEq e e' := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    e : Quiver.Hom u v
    e' : Quiver.Hom u' v'
    ⊢ Iff (Eq (Quiver.Hom.cast hu hv e) e') (HEq e e')
  -/
  rw [Hom.cast_eq_cast]
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    e : Quiver.Hom u v
    e' : Quiver.Hom u' v'
    ⊢ Iff (Eq (_root_.cast ⋯ e) e') (HEq e e')
  -/
  exact _root_.cast_eq_iff_heq
  /-
    🎉 no goals
  -/


theorem Hom.eq_cast_iff_heq {u v u' v' : U} (hu : u = u') (hv : v = v') (e : u ⟶ v) (e' : u' ⟶ v') :
    e' = e.cast hu hv ↔ HEq e' e := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    e : Quiver.Hom u v
    e' : Quiver.Hom u' v'
    ⊢ Iff (Eq e' (Quiver.Hom.cast hu hv e)) (HEq e' e)
  -/
  rw [eq_comm, Hom.cast_eq_iff_heq]
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    e : Quiver.Hom u v
    e' : Quiver.Hom u' v'
    ⊢ Iff (HEq e e') (HEq e' e)
  -/
  exact ⟨HEq.symm, HEq.symm⟩
  /-
    🎉 no goals
  -/


/-- Change the endpoints of a path using equalities. -/
def Path.cast {u v u' v' : U} (hu : u = u') (hv : v = v') (p : Path u v) : Path u' v' :=
  Eq.ndrec (motive := (Path · v')) (Eq.ndrec p hv) hu


theorem Path.cast_eq_cast {u v u' v' : U} (hu : u = u') (hv : v = v') (p : Path u v) :
                                   /-
                                     U : Type u_1
                                     inst✝ : Quiver U
                                     u v u' v' : U
                                     hu : Eq u u'
                                     hv : Eq v v'
                                     p : Quiver.Path u v
                                     ⊢ Eq (Quiver.Path u v) (Quiver.Path u' v')
                                   -/
    p.cast hu hv = _root_.cast (by rw [hu, hv]) p := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    p : Quiver.Path u v
    ⊢ Eq (Quiver.Path.cast hu hv p) (_root_.cast ⋯ p)
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    u' v' : U
    p : Quiver.Path u' v'
    ⊢ Eq (Quiver.Path.cast ⋯ ⋯ p) (_root_.cast ⋯ p)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem Path.cast_rfl_rfl {u v : U} (p : Path u v) : p.cast rfl rfl = p :=
  rfl


@[simp]
theorem Path.cast_cast {u v u' v' u'' v'' : U} (p : Path u v) (hu : u = u') (hv : v = v')
    (hu' : u' = u'') (hv' : v' = v'') :
    (p.cast hu hv).cast hu' hv' = p.cast (hu.trans hu') (hv.trans hv') := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' u'' v'' : U
    p : Quiver.Path u v
    hu : Eq u u'
    hv : Eq v v'
    hu' : Eq u' u''
    hv' : Eq v' v''
    ⊢ Eq (Quiver.Path.cast hu' hv' (Quiver.Path.cast hu hv p)) (Quiver.Path.cast ⋯ …
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    u'' v'' : U
    p : Quiver.Path u'' v''
    ⊢ Eq (Quiver.Path.cast ⋯ ⋯ (Quiver.Path.cast ⋯ ⋯ p)) (Quiver.Path.cast ⋯ ⋯ p)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem Path.cast_nil {u u' : U} (hu : u = u') : (Path.nil : Path u u).cast hu hu = Path.nil := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u u' : U
    hu : Eq u u'
    ⊢ Eq (Quiver.Path.cast hu hu Quiver.Path.nil) Quiver.Path.nil
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    u' : U
    ⊢ Eq (Quiver.Path.cast ⋯ ⋯ Quiver.Path.nil) Quiver.Path.nil
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Path.cast_heq {u v u' v' : U} (hu : u = u') (hv : v = v') (p : Path u v) :
    HEq (p.cast hu hv) p := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    p : Quiver.Path u v
    ⊢ HEq (Quiver.Path.cast hu hv p) p
  -/
  rw [Path.cast_eq_cast]
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    p : Quiver.Path u v
    ⊢ HEq (_root_.cast ⋯ p) p
  -/
  exact _root_.cast_heq _ _
  /-
    🎉 no goals
  -/


theorem Path.cast_eq_iff_heq {u v u' v' : U} (hu : u = u') (hv : v = v') (p : Path u v)
    (p' : Path u' v') : p.cast hu hv = p' ↔ HEq p p' := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    p : Quiver.Path u v
    p' : Quiver.Path u' v'
    ⊢ Iff (Eq (Quiver.Path.cast hu hv p) p') (HEq p p')
  -/
  rw [Path.cast_eq_cast]
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v u' v' : U
    hu : Eq u u'
    hv : Eq v v'
    p : Quiver.Path u v
    p' : Quiver.Path u' v'
    ⊢ Iff (Eq (_root_.cast ⋯ p) p') (HEq p p')
  -/
  exact _root_.cast_eq_iff_heq
  /-
    🎉 no goals
  -/


theorem Path.eq_cast_iff_heq {u v u' v' : U} (hu : u = u') (hv : v = v') (p : Path u v)
    (p' : Path u' v') : p' = p.cast hu hv ↔ HEq p' p :=
  ⟨fun h => ((p.cast_eq_iff_heq hu hv p').1 h.symm).symm, fun h =>
    ((p.cast_eq_iff_heq hu hv p').2 h.symm).symm⟩


theorem Path.cast_cons {u v w u' w' : U} (p : Path u v) (e : v ⟶ w) (hu : u = u') (hw : w = w') :
    (p.cons e).cast hu hw = (p.cast hu rfl).cons (e.cast rfl hw) := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v w u' w' : U
    p : Quiver.Path u v
    e : Quiver.Hom v w
    hu : Eq u u'
    hw : Eq w w'
    ⊢ Eq (Quiver.Path.cast hu hw (p.cons e)) ((Quiver.Path.cast hu ⋯ p).cons (Quiv …
  -/
  subst_vars
  /-
    U : Type u_1
    inst✝ : Quiver U
    v u' w' : U
    p : Quiver.Path u' v
    e : Quiver.Hom v w'
    ⊢ Eq (Quiver.Path.cast ⋯ ⋯ (p.cons e)) ((Quiver.Path.cast ⋯ ⋯ p).cons (Quiver. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem cast_eq_of_cons_eq_cons {u v v' w : U} {p : Path u v} {p' : Path u v'} {e : v ⟶ w}
    {e' : v' ⟶ w} (h : p.cons e = p'.cons e') : p.cast rfl (obj_eq_of_cons_eq_cons h) = p' := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v v' w : U
    p : Quiver.Path u v
    p' : Quiver.Path u v'
    e : Quiver.Hom v w
    e' : Quiver.Hom v' w
    h : Eq (p.cons e) (p'.cons e')
    ⊢ Eq (Quiver.Path.cast ⋯ ⋯ p) p'
  -/
  rw [Path.cast_eq_iff_heq]
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v v' w : U
    p : Quiver.Path u v
    p' : Quiver.Path u v'
    e : Quiver.Hom v w
    e' : Quiver.Hom v' w
    h : Eq (p.cons e) (p'.cons e')
    ⊢ HEq p p'
  -/
  exact heq_of_cons_eq_cons h
  /-
    🎉 no goals
  -/


theorem hom_cast_eq_of_cons_eq_cons {u v v' w : U} {p : Path u v} {p' : Path u v'} {e : v ⟶ w}
    {e' : v' ⟶ w} (h : p.cons e = p'.cons e') : e.cast (obj_eq_of_cons_eq_cons h) rfl = e' := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v v' w : U
    p : Quiver.Path u v
    p' : Quiver.Path u v'
    e : Quiver.Hom v w
    e' : Quiver.Hom v' w
    h : Eq (p.cons e) (p'.cons e')
    ⊢ Eq (Quiver.Hom.cast ⋯ ⋯ e) e'
  -/
  rw [Hom.cast_eq_iff_heq]
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v v' w : U
    p : Quiver.Path u v
    p' : Quiver.Path u v'
    e : Quiver.Hom v w
    e' : Quiver.Hom v' w
    h : Eq (p.cons e) (p'.cons e')
    ⊢ HEq e e'
  -/
  exact hom_heq_of_cons_eq_cons h
  /-
    🎉 no goals
  -/


theorem eq_nil_of_length_zero {u v : U} (p : Path u v) (hzero : p.length = 0) :
    p.cast (eq_of_length_zero p hzero) rfl = Path.nil := by
  /-
    U : Type u_1
    inst✝ : Quiver U
    u v : U
    p : Quiver.Path u v
    hzero : Eq p.length 0
    ⊢ Eq (Quiver.Path.cast ⋯ ⋯ p) Quiver.Path.nil
  -/
  cases p
    /-
      case nil
      U : Type u_1
      inst✝ : Quiver U
      u : U
      hzero : Eq Quiver.Path.nil.length 0
      ⊢ Eq (Quiver.Path.cast ⋯ ⋯ Quiver.Path.nil) Quiver.Path.nil
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      U : Type u_1
      inst✝ : Quiver U
      u v b✝ : U
      a✝¹ : Quiver.Path u b✝
      a✝ : Quiver.Hom b✝ v
      hzero : Eq (a✝¹.cons a✝).length 0
      ⊢ Eq (Quiver.Path.cast ⋯ ⋯ (a✝¹.cons a✝)) Quiver.Path.nil
    -/
  · simp only [Nat.succ_ne_zero, length_cons] at hzero
    /-
      🎉 no goals
    -/


